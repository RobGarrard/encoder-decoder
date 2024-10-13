################################################################################
#
#                                  DATALOADERS
#
################################################################################

import torch

################################################################################
# Datasets


class TranslationDataset(torch.utils.data.Dataset):
    """
    Class for translating data from a 'source' language to a 'target' language.

    Requires 4 files:

        - source_labels_path: File where each line is a source
          sentence (e.g., each line is an English sentence)
        - target_labels_path: File where each line is a target sentence (e.g.,
          each line is a French sentence)
        - source_indices_path: File where each line is a list of indices
          generated by the Language utility class.
        - target_indices_path: File where each line is a list of indices
          generated by the Language utility class.

    Output:
        - name_indices: Tensor of indices for the source language
        - country_indices: Tensor of indices for the target language
        - name: Source sentence
        - country: Target sentence
    """

    def __init__(
        self,
        source_labels_path: str,
        target_labels_path: str,
        source_indices_path: str,
        target_indices_path: str,
    ):
        self.source_labels_path = source_labels_path
        self.target_labels_path = target_labels_path
        self.source_indices_path = source_indices_path
        self.target_indices_path = target_indices_path

        self.source = []
        self.target = []
        self.source_indices = []
        self.target_indices = []

        # Load files
        self._load_data()

        return None

    def _load_label_data(self, path):
        """
        Load in a file where each line is a sentence.
        """
        with open(path, "r") as f:
            data = f.readlines()

        data = [x.strip() for x in data]

        return data

    def _load_index_data(self, path):
        """
        Load in a file where each line is a list of indices.
        """
        with open(path, "r") as f:
            data = f.readlines()

        data = [x.strip().split(" ") for x in data]
        data = [[int(x) for x in y] for y in data]

        return data

    def _load_data(self):
        self.source = self._load_label_data(self.source_labels_path)
        self.target = self._load_label_data(self.target_labels_path)

        self.source_indices = self._load_index_data(self.source_indices_path)
        self.target_indices = self._load_index_data(self.target_indices_path)

        return None

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        """
        # Labels
        source = self.source[idx]
        target = self.target[idx]

        # Indices
        source_indices = self.source_indices[idx]
        target_indices = self.target_indices[idx]

        # Convert to tensors. Note that these are indices, so we use long.
        source_indices = torch.tensor(source_indices).long()
        target_indices = torch.tensor(target_indices).long()

        return source_indices, target_indices, source, target


################################################################################
# Dataloaders


def padding_collator(batch):
    """
    We receive a list of tuples 4 long from the Dataset object (x, y,
    source_label, target_label).

    The xs and ys are sequences of indices. They can be of different lengths!!

    Pad with zeros, which in our Language class corresponds to the <PAD> token.
    """

    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    source = [item[2] for item in batch]
    target = [item[3] for item in batch]

    # Pad the sequences
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)

    return x, y, source, target
