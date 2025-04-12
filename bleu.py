# %% Load model and tokenizer

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu
from itertools import permutations
import json
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from tqdm import tqdm

# %% Utils

def load_dataset_for_lang(lang_code):
    return load_dataset("gsarti/flores_101", lang_code)

def mean(some_list):
    return sum(some_list) / len(some_list)

def save_progress(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

def translate(text, source_lang, target_lang, few_shot_examples, model, tokenizer):
    few_shot_prompt = "\n\n".join([
        f"{example['source']} \\\\ {example['target']}"
        for example in few_shot_examples
    ])
    
    prompt = f"""
        {few_shot_prompt}\n\n{text} \\\\
        
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the newly generated tokens
    new_tokens = outputs[0][input_length:]
    translation = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Only look at first next sentence
    translation = translation.split('\n')[0]
    return translation.strip()

def load_progress(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}

def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Determine output directory based on model name
    output_dir = "bleu_llama" if "llama" in args.model_name.lower() else "bleu_aya"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "translation_results.json")

    N = 2
    languages = {
        "Arabic": "ara", "Chinese": "zho_simpl", "Czech": "ces", "Dutch": "nld",
        "English": "eng", "French": "fra", "German": "deu", "Greek": "ell",
        "Hebrew": "heb", "Hindi": "hin", "Indonesian": "ind", "Italian": "ita",
        "Japanese": "jpn", "Korean": "kor", "Persian": "fas", "Polish": "pol",
        "Portuguese": "por", "Romanian": "ron", "Russian": "rus", "Spanish": "spa",
        "Turkish": "tur", "Ukrainian": "ukr", "Vietnamese": "vie"
    }
    results = load_progress(results_file)
    
    for source_lang, target_lang in permutations(languages.values(), 2):
        if f"{source_lang}-{target_lang}" in results:
            print(f"Skipping {source_lang} to {target_lang} - already processed")
            continue

        print(f"Processing {source_lang} to {target_lang}")
        
        dataset_from = load_dataset_for_lang(source_lang)
        dataset_to = load_dataset_for_lang(target_lang)
        
        reference_bleus = []
        translation_bleus = []
        for i in tqdm(range(128)):
            few_shot_indices = random.sample(list(range(len(dataset_from["devtest"]))), N)
            few_shot_examples = [
                {"source": dataset_from["devtest"][j]["sentence"],
                 "target": dataset_to["devtest"][j]["sentence"]}
                for j in few_shot_indices
            ]

            sentence_from = dataset_from["devtest"][i + N]["sentence"]
            sentence_to = dataset_to["devtest"][i + N]["sentence"]
            
            translation = translate(sentence_from, source_lang, target_lang, few_shot_examples, model, tokenizer)

            if translation:
                reference_bleu = sacrebleu.sentence_bleu(sentence_to, [sentence_to]).score
                translation_bleu = sacrebleu.sentence_bleu(translation, [sentence_to]).score
                
                reference_bleus.append(reference_bleu)
                translation_bleus.append(translation_bleu)
        
        results[f"{source_lang}-{target_lang}"] = {
            "reference_bleu": mean(reference_bleus),
            "translation_bleu": mean(translation_bleus)
        }
        
        print(f"Results for {source_lang} to {target_lang}:")
        print(f"Reference BLEU: {results[f'{source_lang}-{target_lang}']['reference_bleu']}")
        print(f"Translation BLEU: {results[f'{source_lang}-{target_lang}']['translation_bleu']}")
        
        save_progress(results, results_file)

    # Plot results
    languages = set()
    for pair in results.keys():
        src, tgt = pair.split('-')
        languages.update([src, tgt])

    matrix = pd.DataFrame(index=sorted(languages), columns=sorted(languages), data=float('nan'))

    for pair, scores in results.items():
        src, tgt = pair.split('-')
        matrix.loc[src, tgt] = scores['translation_bleu']

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="Blues", fmt=".2f", cbar_kws={'label': 'Translation BLEU'})
    plt.title('Translation BLEU Scores Heatmap')
    plt.xlabel('Target Language')
    plt.ylabel('Source Language')
    plt.savefig(os.path.join(output_dir, "translation_bleu_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BLEU scores for translations")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    args = parser.parse_args()
    main(args)
