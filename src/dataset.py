import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from src.preprocessing import preprocess_audio


class ASVspoof2019Dataset(Dataset):
    """PyTorch Dataset for ASVspoof 2019 LA."""

    def __init__(self, protocol_file: str, audio_dir: str):
        self.audio_dir = audio_dir
        self.samples = []

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                filename = parts[1]
                label = parts[4]
                # bonafide = 0 (real), spoof = 1 (fake)
                label_int = 0 if label == "bonafide" else 1
                self.samples.append((filename, label_int))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        file_path = os.path.join(self.audio_dir, f"{filename}.flac")
        spectrogram = preprocess_audio(file_path)
        # Add channel dimension: (1, n_mels, time_frames)
        spectrogram = spectrogram.unsqueeze(0)
        return spectrogram, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(batch_size: int = 32, num_workers: int = 4):
    """Create train, dev, and eval dataloaders."""
    base = "data/LA"

    train_ds = ASVspoof2019Dataset(
        protocol_file=f"{base}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        audio_dir=f"{base}/ASVspoof2019_LA_train/flac",
    )
    dev_ds = ASVspoof2019Dataset(
        protocol_file=f"{base}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        audio_dir=f"{base}/ASVspoof2019_LA_dev/flac",
    )
    eval_ds = ASVspoof2019Dataset(
        protocol_file=f"{base}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        audio_dir=f"{base}/ASVspoof2019_LA_eval/flac",
    )

    # Weighted sampling for class imbalance in training
    labels = [s[1] for s in train_ds.samples]
    class_counts = [labels.count(0), labels.count(1)]
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, dev_loader, eval_loader, class_counts


if __name__ == "__main__":
    print("Loading datasets...")
    train_loader, dev_loader, eval_loader, class_counts = get_dataloaders(batch_size=8, num_workers=0)

    print(f"\nClass counts (train): bonafide={class_counts[0]}, spoof={class_counts[1]}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Eval batches: {len(eval_loader)}")

    # Test one batch
    print("\nLoading one batch...")
    for specs, labels in train_loader:
        print(f"Batch shape: {specs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels.tolist()}")
        print(f"Spectrogram range: [{specs.min():.2f}, {specs.max():.2f}]")
        break