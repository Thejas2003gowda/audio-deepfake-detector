import torch
import torchaudio
import torchaudio.transforms as T
import os


SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
TARGET_LENGTH = 4 * SAMPLE_RATE  # 4 seconds


def load_audio(file_path: str) -> torch.Tensor:
    """Load FLAC/WAV audio file, resample to 16kHz, return mono waveform."""
    import librosa
    import numpy as np
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return torch.from_numpy(waveform.astype(np.float32))


def pad_or_truncate(waveform: torch.Tensor, target_length: int = TARGET_LENGTH) -> torch.Tensor:
    """Pad short audio with zeros or truncate long audio to fixed length."""
    length = waveform.shape[0]

    if length < target_length:
        # Pad with zeros at the end
        padding = target_length - length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif length > target_length:
        # Truncate to target length (take from start)
        waveform = waveform[:target_length]

    return waveform


def waveform_to_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """Convert waveform to log-mel spectrogram. Output shape: (n_mels, time_frames)."""
    mel_spec = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )(waveform)

    # Convert to log scale (dB)
    log_mel = T.AmplitudeToDB()(mel_spec)

    return log_mel


def preprocess_audio(file_path: str) -> torch.Tensor:
    """Full preprocessing pipeline: load -> pad/truncate -> mel spectrogram."""
    waveform = load_audio(file_path)
    waveform = pad_or_truncate(waveform)
    mel_spec = waveform_to_mel_spectrogram(waveform)
    return mel_spec


if __name__ == "__main__":
    # Test on a real and a fake sample
    import matplotlib.pyplot as plt

    # Find one bonafide and one spoof sample
    train_dir = "data/LA/ASVspoof2019_LA_train/flac"
    protocol_file = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

    bonafide_file = None
    spoof_file = None

    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[1]
            label = parts[4]
            if label == "bonafide" and bonafide_file is None:
                bonafide_file = os.path.join(train_dir, f"{filename}.flac")
            elif label == "spoof" and spoof_file is None:
                spoof_file = os.path.join(train_dir, f"{filename}.flac")
            if bonafide_file and spoof_file:
                break

    print(f"Bonafide sample: {bonafide_file}")
    print(f"Spoof sample: {spoof_file}")

    # Process both
    bonafide_spec = preprocess_audio(bonafide_file)
    spoof_spec = preprocess_audio(spoof_file)

    print(f"\nBonafide spectrogram shape: {bonafide_spec.shape}")
    print(f"Spoof spectrogram shape: {spoof_spec.shape}")
    print(f"Bonafide range: [{bonafide_spec.min():.2f}, {bonafide_spec.max():.2f}]")
    print(f"Spoof range: [{spoof_spec.min():.2f}, {spoof_spec.max():.2f}]")

    # Visualize side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].imshow(bonafide_spec.numpy(), aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Bonafide (Real Speech)")
    axes[0].set_xlabel("Time frames")
    axes[0].set_ylabel("Mel bins")

    axes[1].imshow(spoof_spec.numpy(), aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Spoof (AI-Generated Speech)")
    axes[1].set_xlabel("Time frames")
    axes[1].set_ylabel("Mel bins")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/spectrogram_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved comparison to outputs/spectrogram_comparison.png")