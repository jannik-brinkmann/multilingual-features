import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pyconll
import os


TRACER_KWARGS = {'scan': False, 'validate': False}


def extract_activations(model, dataloader, layer_num):
    """
    Extract activations for the entire dataset.
    """
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting activations"):
            text_batch = batch["sentence"]
            labels = batch["label"]

            with model.trace(text_batch, **TRACER_KWARGS):
                input = model.inputs.save()
                acts = model.model.layers[layer_num].output[0].save()
            
            # Remove padding tokens
            attn_mask = input[1]['attention_mask']
            acts = acts * attn_mask.unsqueeze(-1)

            pooled_acts = acts.sum(1)
            all_activations.append(pooled_acts.float().cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.vstack(all_activations), np.concatenate(all_labels)

def concept_filter(sentence, concept_key, concept_value):
    for token in sentence:
        if concept_key in token.feats and concept_value in token.feats.get(concept_key, {}):
            return True
    return False

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

class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(n_features, 1) 

    def forward(self, x):
        return self.linear(x) 
    
def convert_probe_to_pytorch(probe):
    """Convert sklearn probe to PyTorch model."""
    coef = probe.coef_.ravel()
    bias = probe.intercept_
    torch_probe = LogisticRegressionPyTorch(4096)
    with torch.no_grad():
        torch_probe.linear.weight.copy_(torch.tensor(coef).unsqueeze(0))
        torch_probe.linear.bias.copy_(torch.tensor(bias))
    return torch_probe.to("cuda")