import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

NUM_SEEDS = 50
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_NEW_TOKENS = 1024
MAX_BATCH_SIZE = 256

PROMPT = """
You are a prompt engineer. Your mission is to expand prompts written by user.
You should provide the best prompt for text to image generation in English in 1-2 sentences.
You MUST INCLUDE given word in its original form in a response.
"""

TEMPLATE_PROMPT = lambda homonym: f"""
Expand prompt for this word: "{homonym}". Respond ONLY WITH the example of an expanded prompt, nothing else.
"""

def forward_qwen3_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]

    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for messages in messages_batch
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

    responses = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return [r.strip() for r in responses]

def init_qwen(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        offload_folder="offload",
        attn_implementation="flash_attention_2",
        max_memory={0: "80GB"},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    return model, tokenizer

def expand_prompt(homonyms: list[str]) -> None:
    seeds = list(range(NUM_SEEDS))
    results = []

    model_name = "Qwen3-30B-A3B-Instruct-2507"
    
    print(f"\nLoading {model_name}...")
    model, tokenizer = init_qwen(model_name)

    for seed in tqdm(seeds, desc=f"model {model_name}"):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        prompts = [
            PROMPT + TEMPLATE_PROMPT(homonym)
            for homonym in homonyms
        ]

        all_responses = []
        for i in range(0, len(prompts), MAX_BATCH_SIZE):
            batch_prompts = prompts[i:i + MAX_BATCH_SIZE]
            responses = forward_qwen3_batch(model, tokenizer, batch_prompts)
            all_responses.extend(responses)
            torch.cuda.empty_cache()


        for j, homonym in enumerate(homonyms):
            results.append({
                'seed': seed,
                'homonym': homonym,
                'response': all_responses[j]
            })

        if (seed + 1) % 10 == 0:
            df = pd.DataFrame(results)
            model_dir = model_name.split('/')[-1].replace('-', '_')
            save_dir = os.path.join("generation_prompt_response", model_dir)
            os.makedirs(save_dir, exist_ok=True)
            df.to_csv(os.path.join(save_dir, 'expanded_prompts_temp.csv'), index=False)

        df = pd.DataFrame(results)
        model_dir = model_name.split('/')[-1].replace('-', '_')
        save_dir = os.path.join("generation_prompt_response", model_dir)
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, 'expanded_prompts.csv'), index=False)

if __name__ == '__main__':
    csv_path = '../annotation.tsv'
    homonyms = pd.read_csv(csv_path, sep="\t")["homonym"].unique().tolist()

    expand_prompt(homonyms)
