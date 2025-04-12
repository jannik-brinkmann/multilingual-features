import pyconll
import numpy as np
from torch.utils.data import Dataset
import logging
from sklearn.utils import resample


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
    
    if not positive_samples or not negative_samples:
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