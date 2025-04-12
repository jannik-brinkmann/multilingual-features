import argparse
import glob
import json
import os
from collections import defaultdict

import joblib
import numpy as np
import torch
from functools import partial
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.config import HF_TOKEN
from src.utils import setup_model, setup_autoencoder
from src.probing.data import ProbingDataset, balance_dataset
from src.probing.utils import concept_filter, convert_probe_to_pytorch
from src.utils import get_available_languages, get_available_concepts, load_top_shared_features

TRACER_KWARGS = {'scan': False, 'validate': False}

# Constants
UD_BASE_FOLDER = "./data/universal_dependencies/"
PROBE_DIR = "outputs/probing/probes"
FEATURE_DIR = "outputs/probing/features"
OUTPUT_DIR = "outputs/probing/effects"
AYA_AE_PATH = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model_and_autoencoder(model_name):
    model = LanguageModel(model_name, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)
    submodule = model.model.layers[16]
    if "llama" in model_name:
        autoencoder = setup_autoencoder()
    else:
        autoencoder = setup_autoencoder(checkpoint_path=AYA_AE_PATH)
    return model, submodule, autoencoder

def get_ud_test_filepath(language):
    ud_folder = os.path.join(UD_BASE_FOLDER, f"UD_{language}")
    test_file = glob.glob(os.path.join(ud_folder, "*-ud-test.conllu"))
    return test_file[0] if test_file else None

def load_probe(probe_dir, language, concept_key, concept_value):
    probe_file = os.path.join(probe_dir, f"{language}_{concept_key}_{concept_value}.joblib")
    if not os.path.exists(probe_file):
        return None
    return joblib.load(probe_file)

def load_top_features(feature_dir, concept_key, concept_value, language, k):
    feature_file = os.path.join(feature_dir, f"{concept_key}_{concept_value}.json")
    if not os.path.exists(feature_file):
        return None
    with open(feature_file, 'r') as f:
        data = json.load(f)
    if language not in data:
        return None
    return [feature for feature, _ in data[language]["top_1_percent"][:k]]



def extract_activations(model, batch, layer_num):
    """
    Extract activations for the entire dataset.
    """
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        text_batch = batch["sentence"]
        labels = batch["label"]

        with model.trace(text_batch, **TRACER_KWARGS):
            input = model.inputs.save()
            acts = model.model.layers[layer_num].output[0].save()

        # Remove padding tokens
        attn_mask = input[1]['attention_mask']
        acts = acts * attn_mask.unsqueeze(-1)
        
        all_activations.append(acts.float())
        all_labels.append(labels)
    
    return torch.cat(all_activations, dim=0), torch.cat(all_labels, dim=0)

def evaluate_probe(model, submodule, autoencoder, probe, dataset, top_features=None):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    torch_probe = convert_probe_to_pytorch(probe)
    torch_probe = torch_probe.to(device)
    
    correct = 0
    total = 0
    
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        activations, labels = extract_activations(model, batch, args.layer_num)
        activations = activations.to(device)
        labels = labels.to(device)
        
        if top_features:
            orig_shape = activations.shape
            reshaped_activations = activations.view(-1, activations.size(-1))
            
            encoded = autoencoder.encode(reshaped_activations)
            reconstruction = autoencoder.decode(encoded)
            residual = reshaped_activations - reconstruction

            encoded[:, top_features] = 0
            intervened_activations = autoencoder.decode(encoded) + residual
            
            activations = intervened_activations.view(orig_shape)
        
        pooled_acts = activations.sum(1)
        outputs = torch_probe(pooled_acts)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).long().squeeze()
        
        # Ensure shapes match
        if predictions.shape != labels.shape:
            print(f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}")
            predictions = predictions.view(labels.shape)
        
        # Ensure both tensors are on the same device and have the same dtype
        predictions = predictions.to(device)
        labels = labels.to(device).long()  # Ensure labels are long integers

        correct += (predictions == labels).sum().item()
        total += len(labels)

    accuracy = correct / total if total > 0 else 0
    return accuracy

def main(args):
    model, submodule, autoencoder = setup_model_and_autoencoder(args.model_name)
    
    probe_dir = os.path.join(PROBE_DIR, 'llama' if 'llama' in args.model_name else 'aya')
    feature_dir = os.path.join(FEATURE_DIR, 'llama' if 'llama' in args.model_name else 'aya')
    output_dir = os.path.join(OUTPUT_DIR, 'llama' if 'llama' in args.model_name else 'aya')
    print(f"Probe directory: {probe_dir}")
    print(f"Feature directory: {feature_dir}")
    print(f"Output directory: {output_dir}")
    
    if args.concept_key and args.concept_value:
        concepts = [(args.concept_key, args.concept_value)]
    else:
        concepts = get_available_concepts(probe_dir)
    
    languages = get_available_languages(UD_BASE_FOLDER)
    
    results = defaultdict(lambda: defaultdict(dict))
    
    for concept_key, concept_value in tqdm(concepts, desc="Processing concepts"):
        concept_results = defaultdict(dict)
        for language in languages:
            
            test_filepath = get_ud_test_filepath(language)
            if not test_filepath:
                print(f"Test file not found for {language}. Skipping.")
                continue
            
            probe = load_probe(probe_dir, language, concept_key, concept_value)
            if probe is None:
                print(f"Probe not found for {language}_{concept_key}_{concept_value}. Skipping.")
                continue
            
            filter_criterion = partial(concept_filter, concept_key=concept_key, concept_value=concept_value)
            dataset = ProbingDataset(test_filepath, filter_criterion)
            dataset = balance_dataset(dataset, args.seed)
            
            if dataset is None or len(dataset) < 32:
                print(f"Not enough samples in test set for {language}_{concept_key}_{concept_value}. Skipping.")
                continue
            
            # Baseline evaluation
            baseline_accuracy = evaluate_probe(model, submodule, autoencoder, probe, dataset)
            concept_results[language]["baseline"] = baseline_accuracy
            
            # Evaluation with intervention
            top_features = load_top_shared_features(feature_dir, concept_key, concept_value, languages, args.k)
            if top_features:
                intervention_accuracy = evaluate_probe(model, submodule, autoencoder, probe, dataset, top_features)
                concept_results[language]["intervention"] = intervention_accuracy
        
        # Save results for this concept if there are any
        if concept_results:
            results[f"{concept_key}_{concept_value}"] = concept_results
            save_results(results, output_dir, args)
        else:
            print(f"No results for concept {concept_key}_{concept_value}. Skipping save.")

def save_results(results, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    output_file = get_output_file(output_dir, args)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def get_output_file(output_dir, args):
    if args.concept_key and args.concept_value:
        return os.path.join(output_dir, f"{args.concept_key}_{args.concept_value}_k{args.k}.json")
    else:
        return os.path.join(output_dir, f"k{args.k}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate probes with and without intervention")
    parser.add_argument("--layer_num", type=int, default=16, help="Layer number to extract activations from")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the language model")
    parser.add_argument("--k", type=int, default=128, help="Number of top features to consider for intervention")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--concept_key", type=str, default=None, help="Specific concept key to evaluate (optional)")
    parser.add_argument("--concept_value", type=str, default=None, help="Specific concept value to evaluate (optional)")
    args = parser.parse_args()
    
    main(args)
