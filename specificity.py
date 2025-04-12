import argparse
import json
import os
from collections import defaultdict
import joblib
import torch
import numpy as np
from tqdm import tqdm
from nnsight import LanguageModel

def load_probes(probe_dir, language):
    probes = {}
    for filename in os.listdir(probe_dir):
        if filename.startswith(f"{language}_") and filename.endswith(".joblib"):
            probe_name = filename[len(language)+1:-7]  # Remove language prefix and .joblib suffix
            probe_path = os.path.join(probe_dir, filename)
            probes[probe_name] = joblib.load(probe_path)
    return probes

def extract_activations(model, text, layer_num=16):
    with torch.no_grad():
        with model.trace(text, scan=False, validate=False):
            acts = model.model.layers[layer_num].output[0].save()
        pooled_acts = acts.sum(1)
    return pooled_acts.float().cpu().numpy()

def analyze_translations(input_file, probes, model, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    results = defaultdict(lambda: {'different': 0, 'total': 0})

    for item in tqdm(data['results'], desc="Analyzing translations"):
        original_output = item['original_output']
        intervened_output = item['intervened_output']

        original_activations = extract_activations(model, original_output)
        intervened_activations = extract_activations(model, intervened_output)

        for concept, probe in probes.items():
            original_prediction = probe.predict(original_activations)[0]
            intervened_prediction = probe.predict(intervened_activations)[0]

            results[concept]['total'] += 1
            if original_prediction != intervened_prediction:
                results[concept]['different'] += 1

    # Calculate percentages and average
    percentages = []
    for concept in results:
        total = results[concept]['total']
        different = results[concept]['different']
        percentage = (different / total) * 100 if total > 0 else 0
        results[concept]['percentage'] = percentage
        percentages.append(percentage)

    # Calculate average
    average_percentage = np.mean(percentages)
    results['average'] = {'percentage': average_percentage}

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Average percentage across all probes: {average_percentage:.2f}%")

def main(args):
    # Setup model
    model = LanguageModel(args.model_name, torch_dtype=torch.float16, device_map="auto")

    # Load probes
    probe_dir = f"outputs/probing/probes/{'llama' if 'llama' in args.model_name else 'aya'}"
    probes = load_probes(probe_dir, args.language)

    # Process all relevant files
    input_dir = "counterfactual_translations"
    output_dir = "counterfactual_analysis"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.startswith(f"translation_counterfactuals_results_{args.concept_key}_{args.concept_value}") and filename.endswith(f"to_{args.language}.json"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"analysis_{filename}")
            analyze_translations(input_file, probes, model, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze counterfactual translations")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of the language model")
    parser.add_argument("--language", type=str, default="English", help="Target language to analyze")
    parser.add_argument("--concept_key", type=str, default="Tense", help="Concept key (e.g., 'Gender')")
    parser.add_argument("--concept_value", type=str, default="Past", help="Concept value (e.g., 'Masc')")
    args = parser.parse_args()

    main(args)
