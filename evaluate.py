import torch
import argparse
import transformers
import json
import os
import random
import pickle
import math
import torch.distributed as dist
import numpy as np
import wandb
import torch.nn.functional as F
import bisect
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, TrainingArguments
from transformers import Trainer
from torch.optim.lr_scheduler import LambdaLR
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from typing import List, Optional, Any, NamedTuple, Dict, Tuple
from tqdm import tqdm
from transformers import set_seed
from transformers import AutoModel
from collections import namedtuple

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


class CustomMambaLMHeadModel(MambaLMHeadModel):
    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        
        binary_logits = self.binary_head(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple("CausalLMOutput", ["lm_logits", "binary_logits"])
        return CausalLMOutput(lm_logits=lm_logits, binary_logits=binary_logits)

class SFTDataset(Dataset):
    def __init__(self, data):
        super(SFTDataset, self).__init__()
        assert type(data) == list
        self.input_ids = [entry['input_ids'] for entry in data]
        if 'sentence_indices' not in data[0]:
            raise ValueError('No sentence indices in the evaluation data.')
        else:
            self.sentence_indices = [entry['sentence_indices'] for entry in data]
            self.sentence_labels = [entry['sentence_labels'] for entry in data]
        
        if 'datapoint_id' not in data[0]:
            raise ValueError('No datapoint_id in the evaluation data.')
        else:
            self.datapoint_id = [tokenizer.encode(entry['datapoint_id']) for entry in data]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            'input_ids': self.input_ids[i],
            'sentence_indices': self.sentence_indices[i],
            'sentence_labels': self.sentence_labels[i],
            'datapoint_id': self.datapoint_id[i],
            }

@dataclass
class DataCollatorForSFTDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # Extract all input_ids, sentence_indices, and sentence_labels from instances
        input_ids = [instance['input_ids'] for instance in instances]
        sentence_indices = [instance['sentence_indices'] for instance in instances]
        sentence_labels = [instance['sentence_labels'] for instance in instances]
        datapoint_id = [instance['datapoint_id'] for instance in instances]

        # Pad the input_ids using the tokenizer's padding method
        input_ids_padded = self.tokenizer.pad({
            'input_ids': input_ids
        }, return_tensors='pt')['input_ids']

        datapoint_id_padded = self.tokenizer.pad({
            'input_ids': datapoint_id
        }, return_tensors='pt')['input_ids']

        # Pad sentence_indices and sentence_labels manually as they are not managed by tokenizer
        max_len = max(len(s) for s in sentence_indices)
        sentence_indices_padded = torch.full((len(sentence_indices), max_len), 0)  # 0 is used as a padding index as it is not used elsewhere
        sentence_labels_padded = torch.full((len(sentence_labels), max_len), 0)

        for i in range(len(sentence_indices)):
            sentence_indices_padded[i, :len(sentence_indices[i])] = torch.tensor(sentence_indices[i])
            sentence_labels_padded[i, :len(sentence_labels[i])] = torch.tensor(sentence_labels[i])
        
        return {
            'input_ids': input_ids_padded,
            'sentence_indices': sentence_indices_padded,
            'sentence_labels': sentence_labels_padded,
            'attention_mask': input_ids_padded.ne(self.tokenizer.pad_token_id),
            'datapoint_id': datapoint_id_padded
        }

class SFTDataModule():
    def __init__(self, tokenizer, data: str):
        self.dataset = SFTDataset(data=data)
        self.data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)

class MambaTrainer(Trainer):

    def __init__(self, *args, peak_lr=None, min_lr=None, gradient_accumulation_steps=None, window_size=256000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.world_size = dist.get_world_size()
        self.id2logit = {}
        self.window_size = window_size        
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def create_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.95))
        return self.optimizer
        
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                factor = float(current_step) / float(max(1, num_warmup_steps))
            else:
                total_steps = num_training_steps - num_warmup_steps
                decayed_steps = current_step - num_warmup_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decayed_steps / total_steps))
                decayed_lr = (self.args.learning_rate - self.min_lr) * cosine_decay + self.min_lr
                factor = decayed_lr / self.args.learning_rate
            return factor
        
        warmup_percent = 0.1
        num_warmup_steps = int(warmup_percent * num_training_steps)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False):
        stage = "train" if model.training else "eval"

        input_ids = inputs.pop("input_ids")
        sentence_indices = inputs.pop("sentence_indices")
        sentence_labels = inputs.pop("sentence_labels")
        datapoint_id = inputs.pop("datapoint_id")
        batch_size = input_ids.size(0)

        output = model(input_ids=input_ids)
        binary_logits = output.binary_logits
        sentence_logits = binary_logits.gather(dim=1, index=sentence_indices.unsqueeze(-1).expand(-1, -1, binary_logits.size(-1)))
        sentence_logits = sentence_logits.squeeze(-1)
        mask = (sentence_indices != 0).to(binary_logits.device)
        BCE_loss = torch.tensor(0.0, device=binary_logits.device)
        
        if stage == 'eval':
            with torch.no_grad():
                for i in range(batch_size):
                    dp_id = tuple(datapoint_id[i].tolist())
                    logits = sentence_logits[i]
                    msk = mask[i]
                    valid_logits = logits[msk == 1].cpu()
                    self.id2logit[dp_id] = valid_logits
        

        final_loss = BCE_loss

        return (final_loss, {"logits": sentence_logits}) if return_outputs else final_loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        self.model.eval()
        dist.barrier()
        super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
    def save_model(self, output_dir, _internal_call=None):
        if dist.get_rank() == 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
    
            print('Save model state_dict...')
            torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
            self.tokenizer.save_pretrained(output_dir)
            
            with open(f"{output_dir}/config.json", 'w') as f:
                json.dump(self.model.config.__dict__, f, indent=4)
                
    def _load_from_checkpoint(self, model=None, load_bf16=False, model_path=None):
        # Load model state dict from huggingface checkpoint
        state_dict = AutoModel.from_pretrained(model_path).state_dict()

        if load_bf16:
            state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
        load_result = model.load_state_dict(state_dict, False)
        # release memory
        del state_dict
        self._issue_warnings_after_load(load_result)

def sliding_window(input_ids_without_question, question_ids, end_sentence_indices, window_size):
    if window_size <= 0:
        raise Exception("window size must be larger than the length of question")
    all_slices = []
    end_sentence_indices = [ele-len(question_ids) for ele in end_sentence_indices]
    start_sentence_indices = [0] + [ele+1 for ele in end_sentence_indices[:-1]]

    start_index = 0
    while start_index <= start_sentence_indices[-1]:
        end_index_approximation = start_index + window_size
        end_idx = bisect.bisect_left(end_sentence_indices, end_index_approximation) - 1
        if end_idx == -1:
            end_idx = 0
        if end_sentence_indices[end_idx] <= start_index:
            end_idx += 1
        end_index = end_sentence_indices[end_idx]
        if end_idx == len(end_sentence_indices)-1:
            start_index_approximation = end_index - window_size
            start_idx = bisect.bisect_left(start_sentence_indices, start_index_approximation)
            if start_idx > end_idx:
                start_idx = end_idx
            start_index = start_sentence_indices[start_idx]

            input_slice = input_ids_without_question[start_index:end_index+1]
            all_input_ids = question_ids + input_slice

            sub_start_sentence_indices = start_sentence_indices[start_idx:end_idx+2]
            if (start_sentence_indices[start_idx:end_idx+2] == start_sentence_indices[start_idx:end_idx+1]):
                sub_start_sentence_indices.append(end_sentence_indices[-1]+1)
            sub_start_sentence_indices = [ele-sub_start_sentence_indices[0] for ele in sub_start_sentence_indices][1:]
            sub_end_sentence_indices = [ele-1 for ele in sub_start_sentence_indices]
            sub_end_sentence_indices = [ele+len(question_ids) for ele in sub_end_sentence_indices]
            all_slices.append((all_input_ids, sub_end_sentence_indices, (start_idx, end_idx)))
            break
        else:
            start_idx = start_sentence_indices.index(start_index)

            input_slice = input_ids_without_question[start_index:end_index+1]
            all_input_ids = question_ids + input_slice

            sub_start_sentence_indices = start_sentence_indices[start_idx:end_idx+2]
            if (start_sentence_indices[start_idx:end_idx+2] == start_sentence_indices[start_idx:end_idx+1]):
                sub_start_sentence_indices.append(end_sentence_indices[-1]+1)
            sub_start_sentence_indices = [ele-sub_start_sentence_indices[0] for ele in sub_start_sentence_indices][1:]
            sub_end_sentence_indices = [ele-1 for ele in sub_start_sentence_indices]
            sub_end_sentence_indices = [ele+len(question_ids) for ele in sub_end_sentence_indices]
            all_slices.append((all_input_ids, sub_end_sentence_indices, (start_idx, end_idx)))

            # Move the sliding window
            start_index_approximation = start_index + window_size // 2
            start_idx = bisect.bisect_left(start_sentence_indices, start_index_approximation) - 1
            if start_idx == -1:
                start_idx = 0
            if start_sentence_indices[start_idx] == start_index:
                start_idx += 1
                if start_idx == len(start_sentence_indices):
                    break
            start_index = start_sentence_indices[start_idx]
    return all_slices

def eval_one_dataset(trainer, dataset, window_size, tokenizer, logit_path=None, exp_name=None):
    trainer.id2logit = {}
    agg_id2logit = {}
    batch_size = args.per_device_eval_batch_size
    assert batch_size == 1

    with open(dataset, 'rb') as f:
        raw_data = pickle.load(f)

    data = []
    all_questions = []
    for dataset in raw_data:
        for benchmark_name, datapoint_id in raw_data[dataset]:
            assert dataset == benchmark_name
            cur = raw_data[dataset][(benchmark_name, datapoint_id)]
            if type(datapoint_id) == str:
                assert "*" not in datapoint_id
            cur['datapoint_id'] = f'{benchmark_name}*{datapoint_id}'
            data.append(cur)
            all_questions.append(raw_data[dataset][(dataset, datapoint_id)]["question"])

    question2token = {}
    for question in all_questions:
        question2token[question] = tokenizer.encode(question)

    sub_data = []
    meta_data = {}
    for point in data:
        datapoint_id = point['datapoint_id']
        input_ids = point['input_ids']
        sentence_indices = point['sentence_indices']
        question = point['question']
        if 'sentence_labels' not in point:
            sentence_labels = [0] * len(sentence_indices)
            point['sentence_labels'] = sentence_labels
        else:
            sentence_labels = point['sentence_labels']
        question_ids = question2token[question]
        input_ids_without_question = input_ids[len(question_ids):]
        windows = sliding_window(input_ids_without_question, question_ids, sentence_indices, window_size)
        for index, window in enumerate(windows):
            sub_datapoint_id = datapoint_id + f'*{index}'
            sub_input_ids, sub_sentence_indices, (start_idx, end_idx) = window
            sub_sentence_labels = sentence_labels[start_idx:end_idx+1]
            sub_data.append({"input_ids": sub_input_ids, "sentence_indices": sub_sentence_indices, "datapoint_id": sub_datapoint_id, "sentence_labels": sub_sentence_labels})
            meta_data[(datapoint_id.split("*")[0], datapoint_id.split("*")[1], index)] = (start_idx, end_idx)

    dummy_point = {"input_ids": tokenizer.encode("dummy"), "sentence_indices": [0], "datapoint_id": "dummy", "sentence_labels": [0]}
    num_points = (dist.get_world_size() - (len(sub_data) % dist.get_world_size())) % dist.get_world_size()
    for _ in range(num_points):
        sub_data.append(dummy_point)
    assert len(sub_data) % dist.get_world_size() == 0

    eval_data_module = SFTDataModule(
        tokenizer=tokenizer,
        data=sub_data,
    )

    trainer.eval_dataset = eval_data_module.dataset
    trainer.data_collator  = eval_data_module.data_collator

    trainer.evaluate()

    agg_id2logit = {}

    if dist.get_rank() == 0:
        print('Aggregating id2logit...')

    world_size = dist.get_world_size()
    
    torch.cuda.empty_cache()

    gathered_dicts = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_dicts, trainer.id2logit)

    if dist.get_rank() == 0:
        print('Processing id2logit...')
        for d in gathered_dicts:
            for key, value in d.items():
                detokenized_key = tokenizer.decode(key)
                agg_id2logit[detokenized_key] = value
        
        if "dummy" in agg_id2logit:
            del agg_id2logit["dummy"]
        
        final_agg_id2logit = {}
        for raw_key, value in tqdm(agg_id2logit.items()):
            assert "*" in raw_key
            dataset = raw_key.split("*")[0]
            base_key = raw_key.split("*")[1]
            index = raw_key.split("*")[2]
            if (dataset, base_key) not in final_agg_id2logit:
                final_agg_id2logit[(dataset, base_key)] = {}
            final_agg_id2logit[(dataset, base_key)][int(index)] = value
            
        processed_agg_id2logit_mean = {}
        for key, value in tqdm(final_agg_id2logit.items()):
            all_logits = {}
            for index in value:
                meta_data_key = (key[0], key[1], index)
                if key[0] not in processed_agg_id2logit_mean:
                    processed_agg_id2logit_mean[key[0]] = {}
                cur_logits = value[index] # this is a tensor
                cur_logits = cur_logits.float().cpu().numpy().tolist()
                start_idx, end_idx = meta_data[meta_data_key]
                assert len(cur_logits) == end_idx - start_idx + 1
                for i in range(start_idx, end_idx+1):
                    if i not in all_logits:
                        all_logits[i] = []
                    all_logits[i].append(cur_logits[i-start_idx])
            processed_agg_id2logit_mean[key[0]][key] = [(sum(all_logits[index])/ len(all_logits[index])) for index in range(len(all_logits))]
    
    trainer.id2logit = {}
    
    if dist.get_rank() == 0:
        save_path = f'{logit_path}/{exp_name}.pickle'
        print(f'Saving id2logit to {save_path}...')
        with open(save_path, 'wb') as f:
            pickle.dump(processed_agg_id2logit_mean, f)

def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dist.init_process_group(backend='nccl')
    is_master_node = dist.get_rank() == 0
    wandb.init(project=args.exp_name, name=args.exp_name, config=args)

    model = CustomMambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")
    model.binary_head = torch.nn.Linear(model.config.d_model, 1, dtype=torch.bfloat16)
    
    if "130m" in args.model:
        model_path = "MambaRetriever/mambaretriever-130m"
    elif "1.3b" in args.model:
        model_path = "MambaRetriever/mambaretriever-1.3b"
    else:
        raise ValueError("Model not supported")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    mybf16 = False

    trainer = MambaTrainer(
            model=model,
            tokenizer=tokenizer,
            args=TrainingArguments(
                learning_rate=args.peak_lr,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                optim=args.optim,
                output_dir=f'output/{args.exp_name}',
                save_total_limit=args.save_total_limit,
                logging_dir='./logs',
                logging_steps=args.logging_steps,
                eval_strategy='steps',
                report_to=["wandb"],
                save_strategy=args.save_strategy,
                save_steps=args.save_steps,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                per_device_eval_batch_size = args.per_device_eval_batch_size,
                dataloader_drop_last=True,
                label_names=['sentence_indices', 'sentence_labels', 'datapoint_id'], 
                seed=args.seed,
                fp16=False,
                bf16=mybf16,
            ),
            window_size=args.window_size,
            peak_lr=args.peak_lr,
            min_lr=args.min_lr,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
    
    trainer._load_from_checkpoint(load_bf16=True, model_path=model_path)
        
    assert args.window_size != 1, "Does not support window size of 1. Please set window size to a larger value."
    eval_one_dataset(trainer, args.eval_data_path, args.window_size, tokenizer, logit_path=args.logit_path, exp_name=args.exp_name)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str) 
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--optim", type=str, default="adamw_hf")
    parser.add_argument("--eval_data_path", type=str)
    parser.add_argument("--window_size", type=int, default=256000)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument("--logit_path", type=str)
    parser.add_argument("--peak_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int)
        
    args = parser.parse_args()

    run(args)
