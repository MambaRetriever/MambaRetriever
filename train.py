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
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, TrainingArguments
from transformers import Trainer
from transformers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from typing import List, Optional, Any, NamedTuple, Dict, Tuple
from transformers import set_seed
from torch.utils.data import IterableDataset
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

class SFTStreamingDataset(IterableDataset):
    def __init__(self, input_data_path, tokenizer, eval=False):
        super(SFTStreamingDataset, self).__init__()
        self.input_data_path = input_data_path
        self.tokenizer = tokenizer
        self.eval = eval
        
    def parse_pickle(self):
        with open(self.input_data_path, "rb") as file:
            try:
                while True:
                    data = pickle.load(file)
                    yield data
            except EOFError:
                pass

    def __iter__(self):
        for entry in self.parse_pickle():
            # Extract relevant fields from the current entry
            input_ids = entry['input_ids']
            if 'sentence_indices' not in entry or 'sentence_labels' not in entry or 'input_ids' not in entry:
                continue
            
            sentence_indices = entry['sentence_indices']
            sentence_labels = entry['sentence_labels']
            
            yield {
                'input_ids': input_ids,
                'sentence_indices': sentence_indices,
                'sentence_labels': sentence_labels,
            }

@dataclass
class DataCollatorForSFTDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # Extract all input_ids, sentence_indices, and sentence_labels from instances
        input_ids = [instance['input_ids'] for instance in instances]
        sentence_indices = [instance['sentence_indices'] for instance in instances]
        sentence_labels = [instance['sentence_labels'] for instance in instances]
        
        # Pad the input_ids using the tokenizer's padding method
        input_ids_padded = self.tokenizer.pad({
            'input_ids': input_ids
        }, return_tensors='pt')['input_ids']

        # Pad sentence_indices and sentence_labels manually as they are not managed by tokenizer
        max_len = max(len(s) for s in sentence_indices)
        sentence_indices_padded = torch.full((len(sentence_indices), max_len), 0)  # 0 can be used as a padding index if not used elsewhere
        sentence_labels_padded = torch.full((len(sentence_labels), max_len), 0)

        for i in range(len(sentence_indices)):
            sentence_indices_padded[i, :len(sentence_indices[i])] = torch.tensor(sentence_indices[i])
            sentence_labels_padded[i, :len(sentence_labels[i])] = torch.tensor(sentence_labels[i])
            

        return {
            'input_ids': input_ids_padded,
            'sentence_indices': sentence_indices_padded,
            'sentence_labels': sentence_labels_padded,
            'attention_mask': input_ids_padded.ne(self.tokenizer.pad_token_id),
        }

class SFTDataModule():
    def __init__(self, tokenizer, input_data_path: str):
        self.dataset = SFTStreamingDataset(input_data_path=input_data_path, tokenizer=tokenizer)
        self.data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)

class MambaTrainer(Trainer):

    def __init__(self, *args, peak_lr=None, min_lr=None, gradient_accumulation_steps=None, window_size=256000, prev_train_step = None ,**kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.world_size = dist.get_world_size()
        self.bce_losses = {
            'sum': torch.tensor(0.0, device='cuda'),
            'count': torch.tensor(0, device='cuda')
        }
        self.id2logit = {}
        self.window_size = window_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accumulated_BCE_loss = 0.0
        self.accumulation_steps = 0

    def create_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.95))
        return self.optimizer
        
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                factor = float( current_step) / float(max(1, num_warmup_steps))
            else:
                total_steps = num_training_steps - num_warmup_steps
                decayed_steps = current_step- num_warmup_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decayed_steps / total_steps))
                decayed_lr = (self.args.learning_rate - self.min_lr) * cosine_decay + self.min_lr
                factor = decayed_lr / self.args.learning_rate
            return factor
        
        warmup_percent = 0.1
        num_warmup_steps = int(warmup_percent * num_training_steps)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        return self.lr_scheduler


    def compute_loss(self, model, inputs, return_outputs=False):        
        input_ids = inputs.pop("input_ids")
        sentence_indices = inputs.pop("sentence_indices")
        sentence_labels = inputs.pop("sentence_labels")     
        attention_mask = inputs.pop("attention_mask")
        
        outputs = model(input_ids=input_ids)
        ## calculate binary cross entropy loss
        binary_logits = outputs.binary_logits
        sentence_logits = binary_logits.gather(dim=1, index=sentence_indices.unsqueeze(-1).expand(-1, -1, binary_logits.size(-1)))
        target_labels = sentence_labels.float().to(binary_logits.device)
        sentence_logits = sentence_logits.squeeze(-1)
        mask = (sentence_indices != 0).to(binary_logits.device)
        num_positive_samples = (target_labels == 1).sum()
        num_negative_samples = (sentence_indices!=0).sum() - num_positive_samples
        pos_weight = torch.tensor([num_negative_samples / (num_positive_samples + 1e-10)], device=sentence_logits.device)
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        loss = loss_fct(sentence_logits, target_labels)
        BCE_loss = (loss * mask).sum() / mask.sum() 

        final_loss = BCE_loss
        
        # Accumulate losses
        self.accumulated_BCE_loss += BCE_loss
        self.accumulation_steps += 1
        
        # Accumulate the loss and increment the count
        self.bce_losses['sum'] += BCE_loss.item()
        self.bce_losses['count'] += 1

        # Print and reset after gradient accumulation steps
        if self.accumulation_steps == self.gradient_accumulation_steps:
            # Gather and average across all GPUs
            dist.all_reduce(self.accumulated_BCE_loss, op=dist.ReduceOp.SUM)
            
            average_BCE_loss = self.accumulated_BCE_loss / (self.world_size * self.gradient_accumulation_steps)
            
            if dist.get_rank() == 0:
                print(f'BCE Loss: {average_BCE_loss.item()}')
                    
            # Reset accumulators
            self.accumulated_BCE_loss = 0.0
            self.accumulation_steps = 0

        return (final_loss, {"logits": sentence_logits}) if return_outputs else final_loss


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        dataset_name = None,
        train_stage = None,
    ) -> Dict[str, float]:

        self.model.eval()

        dist.barrier()

        super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Switch back to training mode
        self.model.train()
    
    def save_model(self, output_dir, _internal_call=None):
        if dist.get_rank() == 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
    
            print('Save model state_dict...')
            torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
            self.tokenizer.save_pretrained(output_dir)
            
            with open(f"{output_dir}/config.json", 'w') as f:
                json.dump(self.model.config.__dict__, f, indent=4)


def print_training_args(training_args):
        for key, value in vars(training_args).items():
            print(f"{key}: {value}")

def run(args):
    random.seed(args.seed)     # python random generator
    np.random.seed(args.seed)  # numpy random generator

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    dist.init_process_group(backend='nccl')
    is_master_node = dist.get_rank() == 0

    # Initialize wandb with the correct project name and configuration
    wandb.init(project=args.exp_name, name=args.exp_name, config=args)

    model_checkpoint = args.model

    if is_master_node:
        print("checkpoint: ", model_checkpoint)
 
    if '2.7' in args.model:
        model = CustomMambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")
        model.binary_head = torch.nn.Linear(model.config.d_model, 1, dtype=torch.bfloat16)
    else:
        model = CustomMambaLMHeadModel.from_pretrained(args.model, device="cuda")
        model.binary_head = torch.nn.Linear(model.config.d_model, 1)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token


    data_module = SFTDataModule(
        tokenizer=tokenizer,
        input_data_path=args.data_path,
    )
    
    mybf16 = True
        
    trainer = MambaTrainer(
            model=model,
            train_dataset=data_module.dataset,
            data_collator=data_module.data_collator,
            tokenizer=tokenizer,
            args=TrainingArguments(
                learning_rate=args.peak_lr,
                num_train_epochs=args.num_epochs,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                optim=args.optim,
                output_dir=f'output/{args.exp_name}',
                save_total_limit=args.save_total_limit,
                logging_dir='./logs',
                logging_steps=args.logging_steps,
                eval_strategy='no',
                report_to=["wandb"],
                save_strategy=args.save_strategy,
                save_steps=args.save_steps,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                dataloader_drop_last=True,
                label_names=['sentence_indices', 'sentence_labels'], 
                seed=0,
                fp16=False,
                bf16=mybf16,
            ),
            peak_lr=args.peak_lr,
            min_lr=args.min_lr,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
    
    trainer.eval_results = {}
    
    if args.resume_from_checkpoint == 'True':
        trainer._load_from_checkpoint(resume_from_checkpoint=args.checkpoint_path)
        if is_master_node:
            print(f'Resume Training from Checkpoint in {args.checkpoint_path}')
    
    if is_master_node:
        print('training in progress...')
        print(f'Training on {args.data_path} now...')
    trainer.train()
    
    # Calculate average BCE loss
    bce_loss_stats = trainer.bce_losses
    
    gathered_sum = [torch.zeros_like(bce_loss_stats['sum']) for _ in range(dist.get_world_size())]
    gathered_count = [torch.zeros_like(bce_loss_stats['count']) for _ in range(dist.get_world_size())]
    
    # Gather sums and counts from all GPUs
    dist.all_gather(gathered_sum, bce_loss_stats['sum'])
    dist.all_gather(gathered_count, bce_loss_stats['count'])
    
    # Sum all gathered values across GPUs
    global_sum = torch.stack(gathered_sum).sum()
    global_count = torch.stack(gathered_count).sum()
    
    # Calculate the average BCE loss across all GPUs
    average_bce_loss = global_sum / global_count

    if dist.get_rank() == 0:
        print(f'Average BCE Losses on Synthetic validation set: {average_bce_loss.item()}')

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str, default='EleutherAI/gpt-neox-20b')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--optim", type=str, default='adamw_hf')
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_strategy", type=str, default='steps')
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument("--peak_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5) 
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()

    run(args)
