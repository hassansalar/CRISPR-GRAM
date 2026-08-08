
"""
=============================================================
CRISPR-GRAM Dataset
=============================================================
Pipeline:
Raw Files -> Sequence Parsing -> Biological Feature Extraction
-> Feature Normalization -> PyTorch Dataset

Author : Hassan Salari
=============================================================
"""

import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


class CRISPRDataset(Dataset):
    """
    PyTorch Dataset for CRISPR-GRAM.

    Each sample consists of:
        - aligned sgRNA-target pair
        - binary label
        - 126-dimensional biological feature vector
    """

    def __init__(self, data_dir, data_type="cls", transform=None):

        super().__init__()

        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = transform

        # ----------------------------------------------------
        # nucleotide dictionary
        # ----------------------------------------------------
        self.alphabet = "ACGTN"
        self.token2id = {c: i for i, c in enumerate(self.alphabet)}

        # ----------------------------------------------------
        # load raw files
        # ----------------------------------------------------
        print(f"\nLoading {data_type} dataset ...")

        self.raw_sequences, self.labels = self.load_dataset()

        print(f"Loaded {len(self.raw_sequences):,} samples")

        # ----------------------------------------------------
        # biological feature extraction (needs raw_sequences first!)
        # ----------------------------------------------------
        self.features = self.prepare_features()

        # alias kept for backward-compatible references
        self.combined_features = self.features

        # ----------------------------------------------------
        # normalization (Z-score, fit on this dataset)
        # ----------------------------------------------------
        self.fit_normalizer()

    # ============================================================
    # File loading
    # ============================================================

    def load_dataset(self):
        """
        Read all files inside the dataset directory.
        """

        files = sorted(
            glob.glob(os.path.join(self.data_dir, "**", "*"), recursive=True)
        )

        files = [f for f in files if os.path.isfile(f)]

        sequences = []
        labels = []

        for file in tqdm(files):
            seqs, y = self.parse_file(file)
            sequences.extend(seqs)
            labels.extend(y)

        labels = np.asarray(labels, dtype=np.float32)

        return sequences, labels

    def parse_file(self, filepath):
        """
        Expected format:
            column 5    : sequence
            last column : label
        """

        sequences = []
        labels = []

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split("\t")

                if len(parts) < 5:
                    continue

                sequence = parts[4]
                label = float(parts[-1])

                sequences.append(sequence)
                labels.append(label)

        return sequences, labels

    def __len__(self):
        return len(self.labels)

    # ============================================================
    # Biological Feature Extraction
    # ============================================================

    def one_hot_encode(self, sequence):
        """
        One-hot encoding of nucleotide sequence.
        Shape = (len(sequence), 5)
        """

        encoding = np.zeros((len(sequence), 5), dtype=np.float32)

        for i, nucleotide in enumerate(sequence):
            encoding[i, self.token2id.get(nucleotide, 4)] = 1.0

        return encoding

    def gc_content(self, sequence):
        """
        Compute GC content.
        """

        if len(sequence) == 0:
            return 0.0

        gc = sequence.count("G") + sequence.count("C")

        return gc / len(sequence)

    def energy_proxy(self, sequence):
        """
        Approximate thermodynamic stability using
        dinucleotide composition.
        """

        score = 0.0

        for i in range(len(sequence) - 1):

            pair = sequence[i:i + 2]

            if pair in ["GG", "CC"]:
                score += 2.0
            elif pair in ["GC", "CG"]:
                score += 1.5
            elif pair in ["GA", "GT", "CA", "CT"]:
                score -= 0.5
            else:
                score -= 1.0

        return score / max(len(sequence), 1)

    def split_guide_target(self, sequence):
        """
        Split aligned sequence into sgRNA / target DNA.
        """

        if len(sequence) < 46:
            return sequence, sequence

        guide = sequence[:23]
        target = sequence[23:46]

        return guide, target

    def mismatch_features(self, guide, target):
        """
        Compute mismatch descriptors.
        """

        mismatch_positions = []

        L = min(20, len(guide), len(target))

        for i in range(L):
            if guide[i] != target[i]:
                mismatch_positions.append(i)

        mismatch_count = len(mismatch_positions)

        if mismatch_count == 0:
            pam_distance = 20
        else:
            pam_distance = min([20 - p for p in mismatch_positions])

        return {
            "count": mismatch_count / 20,
            "pam_distance": pam_distance / 20,
            "seed": float(any(p < 8 for p in mismatch_positions)),
            "distal": float(any(p >= 15 for p in mismatch_positions)),
            "ratio": mismatch_count / 20
        }

    def epigenetic_features(self, gc, mismatch_ratio):
        """
        Generate biologically realistic epigenetic proxy features.
        """

        chromatin = 0.30 + 0.50 * gc - 0.20 * mismatch_ratio
        h3k4me3 = 0.40 + 0.30 * (1 - mismatch_ratio) - 0.10 * abs(gc - 0.5)
        methylation = 0.50 - 0.40 * gc + 0.20 * mismatch_ratio
        tf_binding = 0.20 + 0.60 * (1 - mismatch_ratio) - 0.30 * abs(gc - 0.6)

        features = np.array(
            [chromatin, h3k4me3, methylation, tf_binding],
            dtype=np.float32
        )

        return np.clip(features, 0, 1)

    # ============================================================
    # Feature Vector Construction
    # ============================================================

    def build_feature_vector(self, sequence):
        """
        Construct the complete 126-dimensional biological feature vector.
        """

        guide, target = self.split_guide_target(sequence)

        onehot = self.one_hot_encode(sequence).flatten()

        gc = self.gc_content(sequence)
        energy = self.energy_proxy(sequence)

        mismatch = self.mismatch_features(guide, target)

        mismatch_vector = np.array(
            [
                mismatch["count"],
                mismatch["pam_distance"],
                mismatch["seed"],
                mismatch["distal"],
                mismatch["ratio"]
            ],
            dtype=np.float32
        )

        epi = self.epigenetic_features(gc, mismatch["ratio"])

        feature_vector = np.concatenate(
            [
                onehot,
                np.array([gc], dtype=np.float32),
                np.array([energy], dtype=np.float32),
                mismatch_vector,
                epi
            ]
        )

        return feature_vector.astype(np.float32)

    def prepare_features(self):
        """
        Generate biological feature vectors for the entire dataset.
        """

        print("\nExtracting biological features...")

        features = []

        for sequence in tqdm(self.raw_sequences):
            fv = self.build_feature_vector(sequence)
            features.append(fv)

        return np.asarray(features, dtype=np.float32)

    # ============================================================
    # Normalization
    # ============================================================

    def fit_normalizer(self):
        """
        Fit Z-score normalization on this dataset's own features.
        """

        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def transform_features(self, scaler):
        """
        Normalize features using externally-provided (e.g. training) statistics.
        """

        self.scaler = scaler
        self.features = self.scaler.transform(self.features)

    # ============================================================
    # PyTorch Dataset Interface
    # ============================================================

    def __getitem__(self, idx):

        sample = {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
