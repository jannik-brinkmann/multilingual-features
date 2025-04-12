import os
import json
import glob
from collections import defaultdict
import argparse

from src.utils import get_available_languages

UD_BASE_FOLDER = "./data/universal_dependencies/"


def measure_shared_features(feature_dir, k):
    feature_files = glob.glob(os.path.join(feature_dir, '*.json'))
    languages = get_available_languages(UD_BASE_FOLDER)   
    results = {}

    for file_path in feature_files:
        concept = os.path.basename(file_path).replace('.json', '')
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Skip concepts with fewer than two languages
        if len(data) < 2:
            print(f"Skipping {concept}: fewer than two languages")
            continue

        all_features = defaultdict(set)
        for language, lang_data in data.items():
            if language not in languages:
                print(f"Skipping {concept}: language {language} not in available languages")
                continue
            top_k = set([feature for feature, _ in lang_data['top_1_percent'][:k]])
            for feature in top_k:
                all_features[feature].add(language)
        
        shared_features = sum(1 for langs in all_features.values() if len(langs) >= 2)
        total_features = len(all_features)

        if total_features == 0:
            continue
        
        # Compute distribution of feature sharing
        sharing_distribution = defaultdict(int)
        for feature, langs in all_features.items():
            sharing_distribution[len(langs)] += 1
        
        # Convert to percentage and ensure all language counts are represented
        total_features = sum(sharing_distribution.values())
        sharing_distribution = {i: sharing_distribution[i] / total_features 
                                for i in range(1, len(data) + 1)}
        
        results[concept] = {
            'shared_features': shared_features,
            'total_features': total_features,
            'proportion': shared_features / total_features if total_features > 0 else 0,
            'num_languages': len(data),
            'sharing_distribution': sharing_distribution
        }

    return results

def print_results(results):
    print(f"{'Concept':<30} {'Shared Features':<15} {'Total Features':<15} {'Proportion':<10} {'Languages':<10}")
    print("-" * 80)
    for concept, data in results.items():
        print(f"{concept:<30} {data['shared_features']:<15} {data['total_features']:<15} {data['proportion']:.2f} {data['num_languages']:<10}")
        print("Sharing distribution:")
        for num_langs, proportion in data['sharing_distribution'].items():
            print(f"  {num_langs} language{'s' if num_langs > 1 else ''}: {proportion:.2%}")
        print()

def main(args):
    results = measure_shared_features(args.feature_dir, args.k)
    
    if not results:
        print("No concepts with at least two languages found.")
        return


    model_type = 'llama' if 'llama' in args.feature_dir else 'aya'
    output_dir = os.path.join('outputs', 'probing', 'feature_intersection', model_type)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'shared_features_across_concepts_top_{args.k}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure shared features across languages")
    parser.add_argument('--feature_dir', type=str, required=True, help="Directory containing feature JSON files")
    parser.add_argument('--k', type=int, default=32, help="Number of top features to consider")
    args = parser.parse_args()

    main(args)
