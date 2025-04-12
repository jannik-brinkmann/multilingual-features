import os
import json
import logging 
from sklearn.utils import resample
import pyconll
import torch
import glob
from transformers import AutoConfig, AutoTokenizer
from nnsight import LanguageModel
from torch.utils.data import Dataset
from src.autoencoder import GatedAutoEncoder
import numpy as np
from src.config import HF_TOKEN
from collections import defaultdict


def setup_model(model_name_or_path="meta-llama/Meta-Llama-3-8B", device="auto", torch_dtype=torch.float16):
    
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        config=config, 
        padding_side="left",
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LanguageModel(
        model_name_or_path,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device_map=device,
        token=HF_TOKEN
    )
    submodule = model.model.layers[16]
    
    return model, submodule


def setup_autoencoder(checkpoint_path="./checkpoints/autoencoder.pt", device="cuda"):
    dict = GatedAutoEncoder.from_pretrained(checkpoint_path)
    dict.to(device)
    return dict


def get_template_names(template_dir = "./data/templates") -> list[str]:
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".json"):
            template_name = filename.split(".json")[0]
            if template_name not in templates.keys():
                templates[template_name] = []
            with open(os.path.join(template_dir, filename), "r") as f:
                data = json.load(f)
                templates[template_name] = [x for x in data.keys()]
    return templates


def dict_to_json(data, file_path, indent=4):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Open the file in write mode, which will create it if it doesn't exist
    with open(file_path, 'w') as json_file:
        json_file.write('{\n')
        for idx, (key, value) in enumerate(data.items()):
            json_file.write(' ' * indent + json.dumps(key) + ': ')
            if isinstance(value, (dict, list)):
                json_file.write(json.dumps(value, indent=indent))
            else:
                json_file.write(json.dumps(value))
            if idx < len(data) - 1:
                json_file.write(',')
            json_file.write('\n')
        json_file.write('}\n')
    
    print(f"Data successfully written to {file_path}")


def get_features_and_values(conll_file):
    data = pyconll.load_from_file(conll_file)
    features = {}
    for sentence in data:
        for token in sentence:
            for feat, values in token.feats.items():
                if feat not in features:
                    features[feat] = set()
                features[feat].update(values)
    return features

def get_available_languages(ud_base_folder):
    # Load the languages from data/language.json
    with open('data/languages.json', 'r') as f:
        valid_languages = set(json.load(f)["languages"])

    languages = []
    for folder in os.listdir(ud_base_folder):
        if folder.startswith("UD_"):
            language = folder[3:]  # Remove the "UD_" prefix
            if not language.startswith(("Ancient", "Old")) and "-" not in language:
                # Check if the language is in the valid_languages set
                if language in valid_languages:
                    languages.append(language)
    return languages

def get_available_concepts(probe_dir):
    """Get all available concept combinations from probe files that are also in data/concepts.json."""
    # Load valid concepts from data/concepts.json
    with open('data/concepts.json', 'r') as f:
        valid_concepts = json.load(f)

    probe_files = glob.glob(os.path.join(probe_dir, "*.joblib"))
    concepts = set()
    for file in probe_files:
        parts = os.path.basename(file).split('_')
        if len(parts) >= 3:
            concept_key = parts[-2]
            concept_value = parts[-1].replace('.joblib', '')
            
            # Check if the concept is in the valid_concepts dictionary
            if concept_key in valid_concepts and concept_value in valid_concepts[concept_key]:
                concepts.add((concept_key, concept_value))
    
    return list(concepts)


def balance_dataset(dataset, seed):
    """
    Balance the dataset by undersampling the majority class.
    """
    labels = np.array([item['label'] for item in dataset])
    positive_samples = [item for item in dataset if item['label'] == 1]
    negative_samples = [item for item in dataset if item['label'] == 0]
    
    logging.info(f"Before balancing:")
    logging.info(f"  Total samples: {len(dataset)}")
    logging.info(f"  Positive samples: {len(positive_samples)}")
    logging.info(f"  Negative samples: {len(negative_samples)}")
    
    if not positive_samples:
        logging.warning("No positive samples found.")
        return None
    
    if len(positive_samples) < len(negative_samples):
        negative_samples = resample(negative_samples, n_samples=len(positive_samples), random_state=seed)
    else:
        positive_samples = resample(positive_samples, n_samples=len(negative_samples), random_state=seed)
    
    balanced_dataset = positive_samples + negative_samples
    np.random.shuffle(balanced_dataset)
    
    logging.info(f"After balancing:")
    logging.info(f"  Total samples: {len(balanced_dataset)}")
    logging.info(f"  Positive samples: {len(positive_samples)}")
    logging.info(f"  Negative samples: {len(negative_samples)}")
    
    return balanced_dataset

class ProbingDataset(Dataset):
    def __init__(self, conll_file, filter_criterion):
        self.sentences = []
        self.labels = []
        self.filter_criterion = filter_criterion
        self.load_data(conll_file)

    def load_data(self, conll_file):
        data = pyconll.load_from_file(conll_file)
        for sentence in data:
            label = 1 if self.filter_criterion(sentence) else 0
            self.sentences.append(sentence.text)
            self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"sentence": self.sentences[idx], "label": self.labels[idx]}
    
def concept_filter(sentence, concept_key, concept_value):
    for token in sentence:
        if concept_key in token.feats and concept_value in token.feats.get(concept_key, {}):
            return True
    return False

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