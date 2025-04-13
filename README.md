# Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages
 
This repository contains code, data, and links to autoencoders for replicating the experiments of the paper [Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages](https://arxiv.org/abs/2501.06346).

## Setup

**Counterfactual Data.** 
We design datasets consisting of minimal pairs of inputs that differ only with respect to the presence of a grammatical concept. 
For example, we generate counterfactual pairs that elicit singular or plural verbs based on the grammatical number of the subject:

a. The parents near the cars were  
b. The parent near the cars was

This is an adaptation and translation of data from [Arora et al. (2024)](https://arxiv.org/abs/2402.12560). 

**Universal Dependencies.** 
For our experiments, we selected 23 languages from Universal Dependencies 2.1 (UD; [Nivre et al., 2017](https://aclanthology.org/E17-5001/)), a multilingual treebank containing dependency-parsed sentences. 
These correspond to the 23 languages that Aya-23 was trained on. 
The dataset can be downloaded at [Universal Dependencies](https://universaldependencies.org).
Each word in each sentence in UD is annotated with its part of speech and morphosyntactic features, as defined in the UniMorph schema.

**Sparse Autoencoders.** To run experiments with Llama-3-8B or Aya-23-8B, you will need to either train or download sparse autoencoders for each model. You can download dictionaries for [Aya-23-8B](https://huggingface.co/jbrinkma/sae-aya-23-8b-layer16) and [Llama-3-8B](https://huggingface.co/jbrinkma/sae-llama-3-8b-layer16) from HuggingFace. 

## Demo Notebooks

## Citation
If you use any of the code or ideas presented here, please cite our paper:
```bibtex
@misc{brinkmann2025largelanguagemodelsshare,
      title={Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages}, 
      author={Jannik Brinkmann and Chris Wendler and Christian Bartelt and Aaron Mueller},
      year={2025},
      eprint={2501.06346},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.06346}, 
}
```


If you use the dataset, please also cite: 
```bibtex
@inproceedings{arora-etal-2024-causalgym,
    title = "{C}ausal{G}ym: Benchmarking causal interpretability methods on linguistic tasks",
    author = "Arora, Aryaman and Jurafsky, Dan and Potts, Christopher",
    editor = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.785",
    doi = "10.18653/v1/2024.acl-long.785",
    pages = "14638--14663"
}
```

