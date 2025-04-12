import os
import json
import re
import torch

from nnsight import LanguageModel

from src.dataset import Dataset


TRACER_KWARGS = {
    'scan' : False,
    'validate' : False
}


def preprocess_data(
    model: LanguageModel,
    task: str,
    language: str,
    n_samples: int = 32,
    verbose: bool = True
):
        
    # Load the dataset template
    data = Dataset.load_from(task + "/" + language)
    
    # Generate counterfactual examples
    counter = 0
    counterfactuals = []
    while len(counterfactuals) < n_samples:
        counter += 1

        if counter > 128:
            raise Exception("Could not generate enough counterfactuals.")
        
        # Sample a counterfactual
        pair = data.sample_pair()
        clean_prefix, patch_prefix, clean_answer, patch_answer = "".join(pair.base), "".join(pair.src), pair.base_label, pair.src_label
        
        # Tokenize the prefixes and answers
        clean_prefix_tokens = model.tokenizer(
            clean_prefix, 
            return_tensors="pt",
            padding=False).input_ids
        patch_prefix_tokens = model.tokenizer(
            patch_prefix, 
            return_tensors="pt",
            padding=False).input_ids
        clean_answer_tokens = model.tokenizer(
            clean_answer, 
            return_tensors="pt",
            padding=False).input_ids[:, 1:]  # remove BOS token
        patch_answer_tokens = model.tokenizer(
            patch_answer, 
            return_tensors="pt",
            padding=False).input_ids[:, 1:]  # remove BOS token
        
        # Make sure that answers are a single token
        if clean_answer_tokens.shape[1] != 1 or patch_answer_tokens.shape[1] != 1:
            
            # Find the first token that differs between the clean and patch answer
            n_tokens_shortest_answer = min(clean_answer_tokens.shape[1], patch_answer_tokens.shape[1]) - 1
            different_indices = (clean_answer_tokens[:, :n_tokens_shortest_answer] != patch_answer_tokens[:, :n_tokens_shortest_answer]).nonzero(as_tuple=True)
            if different_indices[1].numel() > 0:
                index = different_indices[1][0]
            else:
                index = n_tokens_shortest_answer
                
            # Merge answer tokens that are identical into the prefix 
            clean_prefix_tokens = torch.cat([clean_prefix_tokens, clean_answer_tokens[:, :index]], dim=1)
            patch_prefix_tokens = torch.cat([patch_prefix_tokens, patch_answer_tokens[:, :index]], dim=1)
            
            # Truncate the answers to a single token
            clean_answer_tokens = clean_answer_tokens[:, index:index+1]
            patch_answer_tokens = patch_answer_tokens[:, index:index+1]

        # Skip if the answers are identical
        if clean_answer_tokens == patch_answer_tokens:
            if verbose: print(f"The model predicted the same answer for both the clean and patch prefix.")
            continue
            
        # Skip if the prefixes are not the same length
        if clean_prefix_tokens.shape[1] != patch_prefix_tokens.shape[1]:
            if verbose: print(f"The clean and patch prefixes are not the same length.")
            continue
                
        # Check if the model makes the correct prediction based on the logit difference
        with model.trace(clean_prefix_tokens, **TRACER_KWARGS), torch.no_grad():
            outputs = model.output[0].save()
        diff = outputs[0, -1, clean_answer_tokens] - outputs[0, -1, patch_answer_tokens]
        
        # Skip if the model does not make the right prediction
        if diff < 0: 
            if verbose: print(f"The model does not make the right prediction.")
            continue

        counterfactuals.append({
            "clean_prefix": clean_prefix_tokens,
            "patch_prefix": patch_prefix_tokens,
            "clean_answer": clean_answer_tokens,
            "patch_answer": patch_answer_tokens,
        })
        counter = 0

    return counterfactuals


def preprocess_universal_dependencies(filepath):
    sentences = []
    current_sentence = None
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Start new sentence
            if line.startswith('# text ='):
                if current_sentence:
                    sentences.append(current_sentence)
                text = line.split('=', 1)[1].strip()
                current_sentence = {"text": text, "words": []}
            
            # Extract word information
            elif re.match(r'^\d+\t', line):
                parts = line.split('\t')
                if len(parts) >= 6:
                    word_info = {
                        "word": parts[1],
                        "lemma": parts[2],
                        "pos": parts[3],
                        "features": {}
                    }
                    
                    # Parse features
                    if parts[5] != '_':
                        features = parts[5].split('|')
                        for feature in features:
                            key, value = feature.split('=')
                            word_info["features"][key] = value
                    
                    current_sentence["words"].append(word_info)
            
            # End of file
            elif line == '' and current_sentence:
                sentences.append(current_sentence)
                current_sentence = None

    # Add the last sentence if file doesn't end with a blank line
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def filter_universal_dependencies(ud_data, concept, concept_value=None):
    matching_sentences = []
    
    for sentence in ud_data:
        for word in sentence['words']:
            if concept in word['features']:
                if concept_value:
                    if word['features'][concept] == concept_value:
                        matching_sentences.append(sentence)
                else:
                    matching_sentences.append(sentence)
                break  # Move to the next sentence once we find a match
            
    # Sample an equal amount of examples where concept is not present
    n_samples = len(matching_sentences)
    for sentence in ud_data:
        does_have_concept = False
        for word in sentence['words']:
            if concept in word['features']:
                if concept_value:
                    if word['features'][concept] == concept_value:
                        does_have_concept = True
                else:
                    does_have_concept = True
                    
        if not does_have_concept:
            matching_sentences.append(sentence)
        
        if len(matching_sentences) >= 2 * n_samples:
            break
    
    return matching_sentences
