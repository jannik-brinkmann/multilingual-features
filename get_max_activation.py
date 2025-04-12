import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
import json
from src.utils import setup_autoencoder
from nnsight import LanguageModel

# Constants
NUM_TOKENS = 256_000
BATCH_SIZE = 1
FEATURE_DIM = 32768  # Adjust based on your autoencoder's feature dimension


TRACER_KWARGS = {
    'scan' : False,
    'validate' : False
}

def main(args):
    # Load model and tokenizer
    model = LanguageModel(args.model_name, torch_dtype=torch.float16, device_map="auto")
    submodule = model.model.layers[16]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Setup autoencoder
    autoencoder = setup_autoencoder()
    autoencoder.to("cpu")
    
    # Load Pile dataset
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    
    # Initialize max activations
    max_activations = torch.zeros(FEATURE_DIM, device="cpu")
    
    # Process tokens
    total_tokens = 0
    pbar = tqdm(total=NUM_TOKENS, desc="Processing tokens")
    
    for batch in dataset.iter(batch_size=BATCH_SIZE):
        
        # Get activations
        with torch.no_grad():
            with model.trace(batch["text"][0], **TRACER_KWARGS), torch.inference_mode():
                
                internal_activations = submodule.output[0].save()
        a, b, c = internal_activations.shape
        internal_activations = internal_activations.reshape(a * b, c)
        autoencoder_activations = autoencoder.encode(internal_activations.to("cpu"))
        
        # Update max_activations
        feature_max_values = autoencoder_activations.max(dim=0)[0]
        max_activations = torch.max(max_activations, feature_max_values)
        
        # Update progress
        tokens_in_batch = autoencoder_activations.shape[0]
        total_tokens += tokens_in_batch
        pbar.update(tokens_in_batch)
        
        if total_tokens >= NUM_TOKENS:
            break
    
    pbar.close()
    
    # Create output directory if it doesn't exist
    output_dir = "max_activations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert max activations to a dictionary
    max_activations_dict = {str(i): float(max_activations[i].item()) for i in range(FEATURE_DIM)}
    
    # Save results to JSON file
    model_name_short = args.model_name.split('/')[-1]
    output_file = os.path.join(output_dir, f"max_activations_{model_name_short}.json")
    with open(output_file, 'w') as f:
        json.dump(max_activations_dict, f, indent=2)
    print(f"Max activations saved to {output_file}")
    
    print(f"Total tokens processed: {total_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find max activations for model features")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use (e.g., 'meta-llama/Meta-Llama-3-8B' or 'CohereForAI/aya-23-8B')")
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
