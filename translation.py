import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from einops import rearrange
from src.utils import setup_autoencoder
import random
import argparse
import joblib 
from nnsight import LanguageModel
from src.config import HF_TOKEN
from sacrebleu.metrics import BLEU
from sklearn.metrics import accuracy_score
import pyconll
import sacrebleu
import itertools
from collections import defaultdict
from src.utils import get_available_languages

UD_BASE_FOLDER = "./data/universal_dependencies/"
FEATURE_DIR = "outputs/probing/features"

LANGUAGE_MAPPING = {
    "German": "deu", 
    "English": "eng",
    "French": "fra",
    "Turkish": "tur"
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

CONCEPT_FEATURE_MAPPING = {
    "Gender_Fem": ("Gender", "Fem"),
    "Gender_Masc": ("Gender", "Masc"),
    "Number_Plur": ("Number", "Plur"),
    "Number_Sing": ("Number", "Sing"),
    "Tense_Past": ("Tense", "Past"),
    "Tense_Pres": ("Tense", "Pres"),
}

OPPOSITE_CONCEPTS = {
    "Gender_Fem": "Gender_Masc",
    "Gender_Masc": "Gender_Fem",
    "Number_Plur": "Number_Sing",
    "Number_Sing": "Number_Plur",
    "Tense_Past": "Tense_Pres",
    "Tense_Pres": "Tense_Past",
}

def load_dataset_for_lang(lang_code):
    return load_dataset("gsarti/flores_101", lang_code)

def translate(text, source_lang, target_lang, few_shot_examples, model, tokenizer):
    few_shot_prompt = "\n\n".join([
        f"{example['source']} \\\\ {example['target']}"
        for example in few_shot_examples
    ])
    
    prompt = f"{few_shot_prompt}\n\n{text} \\\\"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    translation = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Only look at first next sentence
    translation = translation.split('\n')[0]
    return translation.strip()

def extract_activations(model, text, layer_num=16):
    with torch.no_grad():
        with model.trace(text, scan=False, validate=False):
            input = model.inputs.save()
            acts = model.model.layers[layer_num].output[0].save()

        # Remove padding tokens
        attn_mask = input[1]['attention_mask']
        acts = acts * attn_mask.unsqueeze(-1)

        pooled_acts = acts.sum(1)
    return pooled_acts.float().cpu().numpy()

def create_intervention_hook(args, autoencoder, activate_features, ablate_features):
    def intervention_hook(module, input, output):
        s = output[0].shape
        batch, seq_len, hidden_size = s[0], s[1], s[2]
        
        int_val = rearrange(output[0], "b seq d_model -> (b seq) d_model")
        f = autoencoder.encode(int_val.float())
        x_hat = autoencoder.decode(f)
        residual = int_val.float() - x_hat
        
        for feature_id in ablate_features:
            f[..., -1, feature_id] = 0.0  # Ablate
        for feature_id in activate_features:
            max_value = max_activations.get(str(feature_id), 0.0)
            f[..., -1, feature_id] = max_value * args.scaling_factor # Activate to maximum observed value
            
        x_hat_intervened = autoencoder.decode(f)
        x_hat_intervened = rearrange(x_hat_intervened, '(b s) h -> b s h', b=batch, s=seq_len)

        reconstruction = residual + x_hat_intervened
        
        output = (reconstruction.half(),) + output[1:]
        return output
    return intervention_hook

def mean(some_list):
    return sum(some_list) / len(some_list)

def calculate_logit_change(probe, original_activations, intervened_activations):
    original_logits = probe.decision_function(original_activations)
    intervened_logits = probe.decision_function(intervened_activations)
    relative_change = (intervened_logits - original_logits) / np.abs(original_logits)
    return relative_change

def load_top_shared_features(feature_dir, concept_key, concept_value, languages, k):
    feature_file = os.path.join(feature_dir, f"{concept_key}_{concept_value}.json")
    if not os.path.exists(feature_file):
        return None
    
    with open(feature_file, 'r') as f:
        data = json.load(f)

    # Collect top features for each language
    feature_counts = defaultdict(int)
    for language in languages:
        if language not in data:
            continue
        top_features = set([feature for feature, _ in data[language]["top_1_percent"][:k]])
        for feature in top_features:
            feature_counts[feature] += 1

    # Select features shared by at least 2 languages
    shared_features = [feature for feature, count in feature_counts.items() if count >= 2]
    shared_features.sort(key=lambda x: feature_counts[x], reverse=True)
    
    return shared_features[:k]  # Return top k shared features

# Load the maximum activations
with open("max_activations/max_activations_Meta-Llama-3-8B.json", "r") as f:
    max_activations = json.load(f)

def main(args):
    # Get all available languages
    languages = get_available_languages(UD_BASE_FOLDER)
    translation_languages = ["English", "French", "German", "Turkish"]
    print(f"Available languages: {languages}")

    # Generate all possible language pairs
    language_pairs = list(itertools.permutations(translation_languages, 2))
    print(f"Total number of language pairs: {len(language_pairs)}")

    # Load probes and get labels
    probe_dir = "outputs/probing/probes/llama"
    probes = {}
    for concept, (concept_key, concept_value) in CONCEPT_FEATURE_MAPPING.items():
        for language in languages:
            if language not in probes:
                probes[language] = {}
            probe_file_pattern = os.path.join(probe_dir, f"{language}_{concept_key}_{concept_value}.joblib")
            matching_files = glob.glob(probe_file_pattern)
            if matching_files:
                probe_file = matching_files[0]
                probes[language][concept] = joblib.load(probe_file)
            else:
                print(f"Warning: No probe file found for {language}_{concept_key}_{concept_value}")
    print("probes", probes.items())

    nnsight_model = LanguageModel(args.model_name, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    autoencoder = setup_autoencoder()

    # Iterate over all concept combinations
    for concept, (concept_key, concept_value) in CONCEPT_FEATURE_MAPPING.items():
        print(f"\nProcessing concept: {concept_key}_{concept_value}")

        # Initialize results dictionary for this concept
        results = {}

        # Initialize the labels dictionary with nested structure
        labels = {lang: {concept: [] for concept in CONCEPT_FEATURE_MAPPING.keys()} for lang in languages}

        # Get top k features for activation and ablation
        k_features = args.num_features
        activate_features = [] # Optional: Manually set features to activate
        feature_dir = os.path.join(FEATURE_DIR, 'llama' if 'llama' in args.model_name else 'aya')
        if not activate_features:
            activate_features = load_top_shared_features(feature_dir, concept_key, concept_value, languages, k=k_features)
        
        # Get the opposite concept for ablation
        opposite_concept = OPPOSITE_CONCEPTS.get(f"{concept_key}_{concept_value}")
        ablate_features = []  # Optional: Manually set features to ablate
        if opposite_concept and not ablate_features:
            opposite_key, opposite_value = opposite_concept.split('_')
            ablate_features = load_top_shared_features(feature_dir, opposite_key, opposite_value, languages, k=k_features)
        else:
            ablate_features = []
        
        if not activate_features and not ablate_features:
            print(f"No features found for {concept_key}_{concept_value} or its opposite. Skipping this concept.")
            continue

        print(f"Activating features: {activate_features}")
        print(f"Ablating features: {ablate_features}")

        # Iterate over all language pairs
        for source_lang, target_lang in tqdm(language_pairs, desc="Processing language pairs"):
            print(f"\nProcessing translation from {source_lang} to {target_lang}")
            
            # Check if the target language has probes for the current concept
            if f"{concept_key}_{concept_value}" not in probes[target_lang]:
                print(f"Skipping {source_lang}-{target_lang} pair: {target_lang} doesn't have a probe for {concept_key}_{concept_value}")
                continue
            
            # Load Flores 101 datasets for this language pair
            dataset_from = load_dataset_for_lang(LANGUAGE_MAPPING[source_lang])["devtest"]
            dataset_to = load_dataset_for_lang(LANGUAGE_MAPPING[target_lang])["devtest"]

            # Setup few-shot examples
            N_few_shot = 2
            few_shot_indices = random.sample(list(range(len(dataset_from))), N_few_shot)
            few_shot_examples = [
                {"source": dataset_from[j]["sentence"],
                 "target": dataset_to[j]["sentence"]}
                for j in few_shot_indices
            ]

            pair_results = []

            # Limit the number of texts to process
            num_texts_to_process = min(args.num_texts, len(dataset_from))

            processed_count = 0
            i = 0

            with tqdm(total=num_texts_to_process, desc="Processing examples") as pbar:
                while processed_count < num_texts_to_process and i < len(dataset_from):
                    input_text = dataset_from[i]["sentence"]
                    target_text = dataset_to[i]["sentence"]
                    
                    # Check if the target concept is not present in the target text
                    target_activations = extract_activations(nnsight_model, target_text)
                    target_label = probes[target_lang][f"{concept_key}_{concept_value}"].predict(target_activations)[0]
                    
                    if target_label == 0:  # Assuming 0 means the concept is not present
                        # Original translation
                        original_output = translate(input_text, source_lang, target_lang, few_shot_examples, hf_model, tokenizer)
                        original_activations = extract_activations(nnsight_model, original_output)
                        original_label = probes[target_lang][f"{concept_key}_{concept_value}"].predict(original_activations)[0]

                        if original_label == 0:
                            # Intervened translations
                            hook = hf_model.model.layers[args.layer_num].register_forward_hook(
                                create_intervention_hook(args, autoencoder, activate_features=activate_features, ablate_features=ablate_features)
                            )
                            intervened_output = translate(input_text, source_lang, target_lang, few_shot_examples, hf_model, tokenizer)
                            hook.remove()
                            intervened_activations = extract_activations(nnsight_model, intervened_output)

                            # Get labels and logits for original and intervened outputs
                            original_labels = {}
                            original_logits = {}
                            intervened_labels = {}
                            intervened_logits = {}

                            for c, probe in probes[target_lang].items():
                                original_labels[c] = original_label
                                original_logits[c] = probe.decision_function(original_activations)[0]
                                intervened_labels[c] = probe.predict(intervened_activations)[0]
                                intervened_logits[c] = probe.decision_function(intervened_activations)[0]

                            pair_results.append({
                                "input": input_text,
                                "original_output": original_output,
                                "intervened_output": intervened_output,
                                "target": target_text,
                                "target_labels": {c: probe.predict(target_activations)[0] for c in CONCEPT_FEATURE_MAPPING.keys()},
                                "original_labels": original_labels,
                                "original_logits": original_logits,
                                "intervened_labels": intervened_labels,
                                "intervened_logits": intervened_logits
                            })
                            
                            processed_count += 1
                            pbar.update(1)
                        else:
                            print(f"Skipping example {i} as the target concept is already present in the translated text.")
                    else:
                            print(f"Skipping example {i} as the target concept is already present in the target text.")
                    
                    i += 1

                if processed_count < num_texts_to_process:
                    print(f"Warning: Only found {processed_count} examples where the target concept was not present. "
                          f"This is less than the requested {num_texts_to_process} examples.")

            # Store results for this language pair
            results[f"{source_lang}-{target_lang}"] = pair_results

            # Optionally, you can save intermediate results after each language pair
            save_results(results, args, concept_key, concept_value)

        # Final save of all results for this concept
        save_results(results, args, concept_key, concept_value)

def save_results(results, args, concept_key, concept_value):
    # Create the 'translations' folder if it doesn't exist
    output_folder = "translations"
    os.makedirs(output_folder, exist_ok=True)

    # Create the output file path
    output_file = os.path.join(output_folder, f"translation_results_{args.model_name.split('/')[-1]}_{concept_key}_{concept_value}.json")

    # Save the results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    
    print(f"Results for {concept_key}_{concept_value} have been written to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run translation counterfactuals")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--num_texts", type=int, default=64, help="Number of texts to process per language pair")
    parser.add_argument("--num_features", type=int, default=1, help="Number of features to use")
    parser.add_argument("--layer_num", type=int, default=16, help="Layer number to intervene on")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="Value to set the feature to")
    args = parser.parse_args()
    main(args)
