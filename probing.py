import argparse
import glob
import logging
import os
from functools import partial

import joblib
import torch
from nnsight import LanguageModel
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import HF_TOKEN
from src.probing.utils import extract_activations, concept_filter, get_available_languages, get_features_and_values
from src.probing.data import ProbingDataset, balance_dataset

# Constants
TRACER_KWARGS = {'scan': False, 'validate': False}
LOG_DIR = 'logs'
UD_BASE_FOLDER = "./data/universal_dependencies/"

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, 'probing.txt'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def setup_model(model_name):
    """Initialize and return the language model."""
    return LanguageModel(model_name, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)

def get_ud_filepaths(language):
    """Get the filepaths for Universal Dependencies train and test files."""
    ud_folder = os.path.join(UD_BASE_FOLDER, f"UD_{language}")
    train_file = glob.glob(os.path.join(ud_folder, "*-ud-train.conllu"))
    test_file = glob.glob(os.path.join(ud_folder, "*-ud-test.conllu"))
    return train_file[0] if train_file else None, test_file[0] if test_file else None

def prepare_datasets(train_filepath, test_filepath, concept_key, concept_value, seed):
    """Prepare and balance the training and test datasets."""
    filter_criterion = partial(concept_filter, concept_key=concept_key, concept_value=concept_value)
    train_dataset = ProbingDataset(train_filepath, filter_criterion)
    test_dataset = ProbingDataset(test_filepath, filter_criterion)

    print("Balancing training dataset...")
    train_dataset = balance_dataset(train_dataset, seed)
    print("Balancing test dataset...")
    test_dataset = balance_dataset(test_dataset, seed)

    return train_dataset, test_dataset

def train_and_evaluate_probe(train_activations, train_labels, test_activations, test_labels, seed):
    """Train a logistic regression probe and evaluate its performance."""
    print("Training logistic regression model...")
    classifier = LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced", solver="newton-cholesky")
    classifier.fit(train_activations, train_labels)

    train_accuracy = classifier.score(train_activations, train_labels)
    test_accuracy = classifier.score(test_activations, test_labels)

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    return classifier

def process_language(args, language):
    """Process a single language for probing."""
    print(f"\nProcessing language: {language}")
    logging.info(f"Processing language: {language}")
    
    model = setup_model(args.model_name)
    train_filepath, test_filepath = get_ud_filepaths(language)

    if not train_filepath or not test_filepath:
        print(f"Skipping {language}: Missing train or test file")
        logging.warning(f"Skipping {language}: Missing train or test file")
        return

    features = get_features_and_values(train_filepath)

    output_dir = f"outputs/probing/probes/{'llama' if 'llama' in args.model_name else 'aya'}"
    for concept_key, values in features.items():
        for concept_value in values:
            print(f"\nProcessing {language} - {concept_key}: {concept_value}")
            logging.info(f"Processing {language} - {concept_key}: {concept_value}")

            model_filename = f"{language}_{concept_key}_{concept_value}.joblib"
            model_path = os.path.join(output_dir, model_filename)
            
            if os.path.exists(model_path):
                print(f"Probe already exists. Skipping.")
                logging.info(f"Probe already exists for {language} - {concept_key}: {concept_value}. Skipping.")
                continue

            train_dataset, test_dataset = prepare_datasets(train_filepath, test_filepath, concept_key, concept_value, args.seed)

            if train_dataset is None or len(train_dataset) < 128 or test_dataset is None:
                print(f"Not enough samples. Skipping.")
                logging.warning(f"Skipping {language} - {concept_key}: {concept_value}: Not enough samples")
                continue

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            print("Extracting activations...")
            train_activations, train_labels = extract_activations(model, train_dataloader, args.layer_num)
            test_activations, test_labels = extract_activations(model, test_dataloader, args.layer_num)

            classifier = train_and_evaluate_probe(train_activations, train_labels, test_activations, test_labels, args.seed)

            os.makedirs(output_dir, exist_ok=True)
            joblib.dump(classifier, model_path)
            print(f"Saved trained model to {model_path}")
            logging.info(f"Saved trained model to {model_path}")

def main(args):
    if args.language:
        languages = [args.language]
        if not os.path.exists(os.path.join(UD_BASE_FOLDER, f"UD_{args.language}")):
            print(f"Error: Language '{args.language}' not found in Universal Dependencies folder.")
            logging.error(f"Language '{args.language}' not found in Universal Dependencies folder.")
            return
    else:
        languages = get_available_languages(UD_BASE_FOLDER)
    
    for language in languages:
        process_language(args, language)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probing script for language models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the language model to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for data loading")
    parser.add_argument("--layer_num", type=int, default=16, help="Layer number to extract activations from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--language", type=str, help="Specific language to process (optional)")
    args = parser.parse_args()

    main(args)