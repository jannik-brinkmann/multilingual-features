# Create 
import torch
import os
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from functools import partial
from baukit import Trace

from collections import defaultdict
import matplotlib.pyplot as plt
from einops import rearrange

from einops import rearrange

from src.loading_utils import preprocess_data


TRACER_KWARGS = {
    'scan' : False,
    'validate' : False
}


def get_task_languages(task):
    languages = []
    data_folder = "./data/"
    task_folder = os.path.join(data_folder, task)
    for filename in os.listdir(task_folder):
        if filename.endswith(".json"):
            languages.append(filename[:-5])
    return languages
            
            
def get_top_k_activations(model, submodule, tokens, autoencoder):
    with model.trace(tokens, **TRACER_KWARGS), torch.inference_mode():
        internal_activations = submodule.output[0].save()
    internal_activations = internal_activations[0, :, :]
    f = autoencoder.encode(internal_activations.view(-1, internal_activations.shape[-1]))
    a = f.topk(autoencoder.k, sorted=False)
    top_acts, top_indices = a[0], a[1]
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (autoencoder.W_dec.mT.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts

def get_autoencoder_activations(model, submodule, tokens, autoencoder):
    with model.trace(tokens, **TRACER_KWARGS), torch.inference_mode():
        internal_activations = submodule.output[0].save()
    internal_activations = internal_activations[0, :, :]
    autoencoder_activations = autoencoder.encode(internal_activations)
    return autoencoder_activations

def get_dictionary_activations(model, dataset, cache_name, max_seq_length, autoencoder, batch_size=32):
    device = model.device
    num_features, d_model = autoencoder.encoder.shape
    datapoints = dataset.num_rows
    dictionary_activations = torch.zeros((datapoints*max_seq_length, num_features))
    token_list = torch.zeros((datapoints*max_seq_length), dtype=torch.int64)
    with torch.no_grad(), dataset.formatted_as("pt"):
        dl = DataLoader(dataset["input_ids"], batch_size=batch_size)
        for i, batch in enumerate(tqdm(dl)):
            batch = batch.to(device)
            token_list[i*batch_size*max_seq_length:(i+1)*batch_size*max_seq_length] = rearrange(batch, "b s -> (b s)")
            with Trace(model, cache_name) as ret:
                _ = model(batch).logits
                internal_activations = ret.output
                # check if instance tuple
                if(isinstance(internal_activations, tuple)):
                    internal_activations = internal_activations[0]
            batched_neuron_activations = rearrange(internal_activations, "b s n -> (b s) n" )
            batched_dictionary_activations = autoencoder.encode(batched_neuron_activations)
            dictionary_activations[i*batch_size*max_seq_length:(i+1)*batch_size*max_seq_length,:] = batched_dictionary_activations.cpu()
    return dictionary_activations, token_list

def download_dataset(dataset_name, tokenizer, max_length=256, num_datapoints=None):
    if(num_datapoints):
        split_text = f"train[:{num_datapoints}]"
    else:
        split_text = "train"
    dataset = load_dataset(dataset_name, split=split_text).map(
        lambda x: tokenizer(x['text']),
        batched=True,
    ).filter(
        lambda x: len(x['input_ids']) > max_length
    ).map(
        lambda x: {'input_ids': x['input_ids'][:max_length]}
    )
    return dataset

def make_colorbar(min_value, max_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    # Add color bar
    colorbar = ""
    num_colors = 4
    if(min_value < -negative_threshold):
        for i in range(num_colors, 0, -1):
            ratio = i / (num_colors)
            value = round((min_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1); color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    # Do zero
    colorbar += f'<span style="background-color:rgba({white},{white},{white},1);color:rgb(0,0,0)">&nbsp0.0&nbsp</span>'
    # Do positive
    if(max_value > positive_threshold):
        for i in range(1, num_colors+1):
            ratio = i / (num_colors)
            value = round((max_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1);color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    return colorbar

def value_to_color(activation, max_value, min_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    if activation > positive_threshold:
        ratio = activation/max_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)'
    elif activation < -negative_threshold:
        ratio = activation/min_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1)'
    else:
        text_color = "0,0,0"
        background_color = f'rgba({white},{white},{white},1)'
    return text_color, background_color

def convert_token_array_to_list(array):
    print(type(array))
    if isinstance(array, torch.Tensor):
        if array.dim() == 1:
            array = [array.tolist()]
        elif array.dim()==2:
            array = array.tolist()
        else: 
            raise NotImplementedError("tokens must be 1 or 2 dimensional")
    elif isinstance(array, list):
        # ensure it's a list of lists
        if isinstance(array[0], int):
            array = [array]
    return array

def tokens_and_activations_to_html(toks, activations, tokenizer, logit_diffs=None, model_type="causal"):
    # text_spacing = "0.07em"
    text_spacing = "0.00em"
    toks = convert_token_array_to_list(toks)
    activations = convert_token_array_to_list(activations)
    # toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '↵') for t in tok] for tok in toks]
    toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '\\n') for t in tok] for tok in toks]
    highlighted_text = []
    # Make background black
    # highlighted_text.append('<body style="background-color:black; color: white;">')
    highlighted_text.append("""
<body style="background-color: black; color: white;">
""")
    max_value = max([max(activ) for activ in activations])
    min_value = min([min(activ) for activ in activations])
    if(logit_diffs is not None and model_type != "reward_model"):
        logit_max_value = max([max(activ) for activ in logit_diffs])
        logit_min_value = min([min(activ) for activ in logit_diffs])

    # Add color bar
    highlighted_text.append("Token Activations: " + make_colorbar(min_value, max_value))
    if(logit_diffs is not None and model_type != "reward_model"):
        highlighted_text.append('<div style="margin-top: 0.1em;"></div>')
        highlighted_text.append("Logit Diff: " + make_colorbar(logit_min_value, logit_max_value))
    
    highlighted_text.append('<div style="margin-top: 0.5em;"></div>')
    for seq_ind, (act, tok) in enumerate(zip(activations, toks)):
        for act_ind, (a, t) in enumerate(zip(act, tok)):
            if(logit_diffs is not None and model_type != "reward_model"):
                highlighted_text.append('<div style="display: inline-block;">')
            text_color, background_color = value_to_color(a, max_value, min_value)
            highlighted_text.append(f'<span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{t.replace(" ", "&nbsp")}</span>')
            if(logit_diffs is not None and model_type != "reward_model"):
                logit_diffs_act = logit_diffs[seq_ind][act_ind]
                _, logit_background_color = value_to_color(logit_diffs_act, logit_max_value, logit_min_value)
                highlighted_text.append(f'<div style="display: block; margin-right: {text_spacing}; height: 10px; background-color:{logit_background_color}; text-align: center;"></div></div>')
        if(logit_diffs is not None and model_type=="reward_model"):
            reward_change = logit_diffs[seq_ind].item()
            text_color, background_color = value_to_color(reward_change, 10, -10)
            highlighted_text.append(f'<br><span>Reward: </span><span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{reward_change:.2f}</span>')
        highlighted_text.append('<div style="margin-top: 0.2em;"></div>')
        # highlighted_text.append('<br><br>')
    # highlighted_text.append('</body>')
    highlighted_text = ''.join(highlighted_text)
    return highlighted_text

def get_autoencoder_activation(model, cache_name, tokens, autoencoder):
    device = model.device
    with Trace(model, cache_name) as ret, torch.no_grad():
        _ = model(tokens.to(device))
        internal_activations = ret.output
        # check if instance tuple
        if(isinstance(internal_activations, tuple)):
            internal_activations = internal_activations[0]
    internal_activations = rearrange(internal_activations, "b s n -> (b s) n" )
    autoencoder_activations = autoencoder.encode(internal_activations)
    return autoencoder_activations

def ablate_context_one_token_at_a_time(model, dataset, cache_name, autoencoder, feature, max_ablation_length=20):
    all_changed_activations = []
    for token_ind, token_l in enumerate(dataset):
    # for token_ind, token_l in enumerate(full_token_list):
        seq_size = len(token_l)
        original_activation = get_autoencoder_activation(model, cache_name, torch.tensor(token_l).unsqueeze(0), autoencoder)
        original_activation = original_activation[-1,feature].item()
        # Run through the model for each seq length
        if(seq_size==1):
            continue # Size 1 sequences don't have any context to ablate
        # changed_activations = torch.zeros(seq_size).cpu() + original_activation
        changed_activations = torch.zeros(seq_size).cpu()
        minimum_seq_length = max(1, seq_size-max_ablation_length)
        for i in range(minimum_seq_length, seq_size-1):
            ablated_tokens = token_l[:i] + token_l[i+1:]
            # ablated_tokens = token_l
            ablated_tokens = torch.tensor(ablated_tokens).unsqueeze(0)
            with torch.no_grad():
                dictionary_activations = get_autoencoder_activation(model, cache_name, ablated_tokens, autoencoder)
                changed_activations[i] = dictionary_activations[-1,feature].item()
        # changed_activations -= original_activation
        changed_activations[minimum_seq_length:] -= original_activation
        all_changed_activations.append(changed_activations.tolist())
    return all_changed_activations

def save_token_display(tokens, activations, tokenizer, path, save=True, logit_diffs=None, show=False, model_type="causal"):
    html = tokens_and_activations_to_html(tokens, activations, tokenizer, logit_diffs, model_type=model_type)
    return display(HTML(html))

def get_feature_indices(feature_index, dictionary_activations, k=10, setting="max"):
    best_feature_activations = dictionary_activations[:, feature_index]
    # Sort the features by activation, get the indices
    if setting=="max":
        found_indices = torch.argsort(best_feature_activations, descending=True)[:k]
    elif setting=="uniform":
        # min_value = torch.min(best_feature_activations)
        min_value = torch.min(best_feature_activations)
        max_value = torch.max(best_feature_activations)

        # Define the number of bins
        num_bins = k

        # Calculate the bin boundaries as linear interpolation between min and max
        bin_boundaries = torch.linspace(min_value, max_value, num_bins + 1)

        # Assign each activation to its respective bin
        bins = torch.bucketize(best_feature_activations, bin_boundaries)

        # Initialize a list to store the sampled indices
        sampled_indices = []

        # Sample from each bin
        for bin_idx in torch.unique(bins):
            if(bin_idx==0): # Skip the first one. This is below the median
                continue
            # Get the indices corresponding to the current bin
            bin_indices = torch.nonzero(bins == bin_idx, as_tuple=False).squeeze(dim=1)
            
            # Randomly sample from the current bin
            sampled_indices.extend(np.random.choice(bin_indices, size=1, replace=False))

        # Convert the sampled indices to a PyTorch tensor & reverse order
        found_indices = torch.tensor(sampled_indices).long().flip(dims=[0])
    else: # random
        # get nonzero indices
        nonzero_indices = torch.nonzero(best_feature_activations)[:, 0]
        # shuffle
        shuffled_indices = nonzero_indices[torch.randperm(nonzero_indices.shape[0])]
        found_indices = shuffled_indices[:k]
    return found_indices

def get_feature_datapoints(found_indices, best_feature_activations, tokenizer, max_seq_length, dataset):
    num_datapoints = dataset.num_rows
    datapoint_indices =[np.unravel_index(i, (num_datapoints, max_seq_length)) for i in found_indices]
    all_activations = best_feature_activations.reshape(num_datapoints, max_seq_length).tolist()
    full_activations = []
    partial_activations = []
    text_list = []
    full_text = []
    token_list = []
    full_token_list = []
    for i, (md, s_ind) in enumerate(datapoint_indices):
        md = int(md)
        s_ind = int(s_ind)
        full_tok = torch.tensor(dataset[md]["input_ids"])
        full_text.append(tokenizer.decode(full_tok))
        tok = dataset[md]["input_ids"][:s_ind+1]
        full_activations.append(all_activations[md])
        partial_activations.append(all_activations[md][:s_ind+1])
        text = tokenizer.decode(tok)
        text_list.append(text)
        token_list.append(tok)
        full_token_list.append(full_tok)
    return text_list, full_text, token_list, full_token_list, partial_activations, full_activations


def get_token_statistics(feature, feature_activation, dataset, tokenizer, max_seq_length, tokens_for_each_datapoint, save_location="", num_unique_tokens=10, setting="input", negative_threshold=-0.01):
    if(setting=="input"):
        nonzero_indices = feature_activation.nonzero()[:, 0]  # Get the nonzero indices
    else:
        nonzero_indices = (feature_activation < negative_threshold).nonzero()[:, 0]
    nonzero_values = feature_activation[nonzero_indices].abs()  # Get the nonzero values

    # Unravel the indices to get the token IDs
    datapoint_indices = [np.unravel_index(i, (dataset.num_rows, max_seq_length)) for i in nonzero_indices]
    all_tokens = [dataset[int(md)]["input_ids"][int(s_ind)] for md, s_ind in datapoint_indices]

    # Find the max value for each unique token
    token_value_dict = defaultdict(int)
    for token, value in zip(all_tokens, nonzero_values):
        token_value_dict[token] = max(token_value_dict[token], value)
    # if(setting=="input"):
    sorted_tokens = sorted(token_value_dict.keys(), key=lambda x: -token_value_dict[x])
    # else:
    #     sorted_tokens = sorted(token_value_dict.keys(), key=lambda x: token_value_dict[x])
    # Take the top 10 (or fewer if there aren't 10)
    max_tokens = sorted_tokens[:min(num_unique_tokens, len(sorted_tokens))]
    total_sums = nonzero_values.abs().sum()
    max_token_sums = []
    token_activations = []
    assert len(max_tokens) > 0, "No tokens found for this feature"
    for max_token in max_tokens:
        # Find ind of max token
        max_token_indices = tokens_for_each_datapoint[nonzero_indices] == max_token
        # Grab the values for those indices
        max_token_values = nonzero_values[max_token_indices]
        max_token_sum = max_token_values.abs().sum()
        max_token_sums.append(max_token_sum)
        token_activations.append(max_token_values)

    if(setting=="input"):
        title_text = "Input Token Activations"
        save_name = "input"
        y_label = "Feature Activation"
    else:
        title_text = "Output Logit-Difference"
        save_name = "logit_diff"
        y_label = "Logit Difference"

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Add a supreme title for the entire figure
    if(setting=="input"):
        fig.suptitle(f"Feature: {feature}", fontsize=32)

    # Boxplot on the left
    ax = axs[0]
    ax.set_title(f'{title_text}')
    max_text = [tokenizer.decode([t]).replace("\n", "\\n").replace(" ", "_") for t in max_tokens]
    ax.set_ylabel(y_label)
    plt.sca(ax)
    plt.xticks(rotation=35)
    ax.title.set_size(24)
    ax.yaxis.label.set_size(20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.boxplot(token_activations[::-1], labels=max_text[::-1])

    # Bar graph on the right
    ax = axs[1]
    ax.set_title(f'Weighted % of {title_text}')
    plt.sca(ax)
    plt.xticks(rotation=35)
    ax.title.set_size(24)
    ax.yaxis.label.set_size(20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel(f'Weighted Percentage of Total {y_label}')
    ax.bar(max_text[::-1], [t/total_sums*100 for t in max_token_sums[::-1]])

    # Save the figure
    plt.savefig(f'{save_location}feature_{feature}_{save_name}_combined.png', bbox_inches='tight')

    return
