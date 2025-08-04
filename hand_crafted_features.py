import numpy as np
import os
import matplotlib.pyplot as plt

from liverpool.closeset.dataset_preparation import LoadDataset, awgn


# Load your dataset (assuming data is already available as NumPy arrays)
# data: (30000, 8192), label: (30000,)
# Each device has 1000 signals (i.e., label 0 has signals 0-999, label 1 has signals 1000-1999, etc.)

# Example: data.shape = (30000, 8192)
# Example: label.shape = (30000,)
def process_labels(labels):
    return labels.flatten() if labels.shape[1] == 1 else labels


def compute_rf_features(signal, sampling_rate=1e6):
    """
    Compute IQ Imbalance (Amplitude & Phase), CFO, and LO Leakage for a given IQ signal.
    Args:
        signal (np.array): Complex IQ samples of shape (8192,)
        sampling_rate (float): Sampling rate of the signal in Hz (default 1 MHz)
    Returns:
        tuple: (amplitude imbalance, phase imbalance, CFO, LO leakage)
    """
    N = len(signal)

    # Separate I and Q
    I = np.real(signal)
    Q = np.imag(signal)

    # **1. Amplitude Imbalance**
    rms_I = np.sqrt(np.mean(I ** 2))
    rms_Q = np.sqrt(np.mean(Q ** 2))
    amp_imbalance = np.abs(rms_I - rms_Q) / (rms_I + rms_Q)

    # **2. Phase Imbalance**
    phase_imbalance = np.arctan2(np.mean(Q), np.mean(I))

    # **3. Carrier Frequency Offset (CFO)**
    phase_diff = np.angle(signal[1:] * np.conj(signal[:-1]))  # Phase difference between consecutive samples
    mean_phase_diff = np.mean(phase_diff)
    CFO = mean_phase_diff / (2 * np.pi * (1 / sampling_rate))  # Convert to Hz

    # **4. LO Leakage (DC component)**
    LO_leakage = np.abs(np.mean(signal))  # Mean of the complex samples

    return amp_imbalance, phase_imbalance, CFO, LO_leakage


def process_and_plot_histograms(data, labels, output_dir="rf_feature_histograms"):
    """
    Compute RF parameters for all signals, create histograms, and save them in labeled folders.
    Args:
        data (np.array): IQ data of shape (30000, 8192)
        labels (np.array): Label array of shape (30000,)
        output_dir (str): Directory to save histograms
    """
    # Flatten labels if necessary
    labels = process_labels(labels)

    unique_labels = np.unique(labels)

    # Dictionary to store results per label
    rf_features = {label: {"amp_imbalance": [], "phase_imbalance": [], "CFO": [], "LO_leakage": []} for label in
                   unique_labels}

    # Compute RF parameters for each signal
    for i, signal in enumerate(data):
        amp_imb, phase_imb, CFO, LO_leak = compute_rf_features(signal)
        label = labels[i]
        rf_features[label]["amp_imbalance"].append(amp_imb)
        rf_features[label]["phase_imbalance"].append(phase_imb)
        rf_features[label]["CFO"].append(CFO)
        rf_features[label]["LO_leakage"].append(LO_leak)

    # Define folders for each parameter
    param_folders = ["amp_imbalance", "phase_imbalance", "CFO", "LO_leakage"]
    for folder in param_folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # Create histograms for each parameter and save in the respective folder
    for feature in param_folders:
        for label in unique_labels:
            plt.figure(figsize=(8, 6))
            plt.hist(rf_features[label][feature], bins=50, alpha=0.75, edgecolor="black")
            plt.xlabel(feature.replace("_", " ").title())
            plt.ylabel("Number of Signals")
            plt.title(f"{feature.replace('_', ' ').title()} Histogram for Device {label}")
            plt.grid(True)

            # Save the histogram with label name
            filename = f"{feature}_Device_{label}.png"
            plt.savefig(os.path.join(output_dir, feature, filename))
            plt.close()

    print(f"Histograms saved in {output_dir}/")


if __name__ == "__main__":
    file_path_in = 'C:/Users/omer1/Desktop/studies/thesis/LoRa_RFFI_dataset/dataset/Train/dataset_training_aug.h5'
    dev_range = range(0, 30)
    pkt_range = range(0, 1000)
    LoadDatasetObj = LoadDataset()
    data_train, label_train = LoadDatasetObj.load_iq_samples(file_path=file_path_in,
                                                             dev_range=dev_range,
                                                             pkt_range=pkt_range)



    # Add noise to increase system robustness
    data_train = awgn(data_train, range(20, 80))
    process_and_plot_histograms(data_train, label_train, output_dir="C:/Users/omer1/Desktop/studies/thesis"
                                                                    "/hand_crafted_features/histograms")
