from openai import OpenAI
from .util import jpg_path2base64
import httpx



openai_api_key = "REPLACE WITH YOUR OPENAI API KEY"

def gpt_generation(prompt,model,temperature=0,max_tokens=1024,top_p=1,frequency_penalty=0,presence_penalty=0):
    client = OpenAI(api_key = openai_api_key, timeout=httpx.Timeout(60.0, read=20.0, write=40.0, connect=6.0))
    conversation = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(
      model=model,
      messages=conversation,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
    )
    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)


def gpt_generation_with_image(prompt, model, temperature=0, max_tokens=1024, top_p=1, frequency_penalty=0, presence_penalty=0):

    if 'vision' not in model and 'gpt-4-turbo-2024-04-09' not in model and 'gpt-4-turbo' not in model and 'gpt-4o' not in model:
        raise ValueError("""The current OpenAI vision models are ['gpt-4-vision-preview', 'gpt-4-1106-vision-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4o', ...] Go check https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4 for more details.""")

    client = OpenAI(api_key = openai_api_key)
    content_list = []
    content_list.append({"type": "text", "text": prompt['prompt_text']})

    for img_path in prompt['images']:
        content_list.append({
            "type": "image_url",
        	"image_url": {
                "url": f"data:image/jpeg;base64,{jpg_path2base64(img_path)}",
                "detail": "high"
            },
        })

    response = client.chat.completions.create(model=model,
      messages=[
        {
          "role": "user",
          "content": content_list,
        }
      ],
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty
    )

    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)



def gpt_generation_multi_turn(prompt,model,temperature=0,max_tokens=1024,top_p=1,frequency_penalty=0,presence_penalty=0):
    if type(prompt) is not dict or 'conversation' not in prompt.keys():
        raise ValueError("For multiple turn case, the prompt dict provided is not in correct format")
    client = OpenAI(api_key = openai_api_key)
    conversation = prompt['conversation']
    response = client.chat.completions.create(
      model=model,
      messages=conversation,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty
    )
    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)