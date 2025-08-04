import os
import sys
print("start")
import numpy as np
print("import numpy as np done")
# Get the absolute path of the directory containing the current script.
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go two levels up to reach the "hazanom" folder.
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
# Print for debugging (optional)
print("Current directory:", current_dir)
print("parent directory:", parent_dir)

# Add the grandparent directory to sys.path if it's not already there.
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam


import dataset_preparation
from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram

import deep_learning_models_pytorch
#from deep_learning_models_pytorch import TripletNet, identity_loss, TripletNet_hcf
from deep_learning_models_pytorch import TripletNet, TripletNet_hcf, MahalanobisTripletLoss, estimate_class_mean_cov

# from playground import feature_representation

from hand_crafted_features import compute_rf_features
from models import VQVAE, VQVAE_s, VQVAE_no_q, VQVAE_s_no_q, VQVAE_s_no_q_hcf,VQVAE_s_hcf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import silhouette_score


num_epochs = 500

class RFSingleDataset(Dataset):
    """Return (spectrogram, extra_features, label) per sample."""
    def __init__(self, spectrogram, extra, labels):
        self.spectrogram = spectrogram
        self.extra       = extra
        self.labels      = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.spectrogram[idx],
            self.extra[idx],
            self.labels[idx],
        )

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate false alarm probability, miss detection probability, and accuracy.
    """
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # Calculate metrics
    P_FA = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Alarm Probability
    P_MD = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss Detection Probability
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return P_FA, P_MD, accuracy

def train_feature_extractor(
    file_path="/sise/home/talkoz/omer_model_based/LoRa_RFFI_dataset/dataset/Train/dataset_training_aug.h5",
    dev_range=np.arange(0, 30, dtype=int),
    pkt_range=np.arange(0, 1000, dtype=int),
    snr_range=np.arange(20, 80),
    best_model_path="/sise/home/talkoz/omer_model_based/liverpool/openset/vq_vae_models/triplet_net/",
    pretrained_model_path=None  # <-- NEW ARG
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    train_feature_extractor trains an RFF extractor using triplet loss,
    with an option to load a pretrained model and copy over only its encoder.

    file_path:       path to training dataset
    dev_range:       label range of devices for training
    pkt_range:       packet range from each device for training
    snr_range:       SNR range used for data augmentation
    best_model_path: directory to save the best performing model
    pretrained_model_path: optional path to a PyTorch model file that has
                           .encoder and .classifier. Only .encoder is copied
                           into our TripletNet's encoder.

    Returns:
        Nothing directly, but saves out best_model.pth and prints progress.
    """

    LoadDatasetObj = LoadDataset()

    # 1. Load preamble IQ samples and labels.
    data, label = LoadDatasetObj.load_iq_samples(file_path, dev_range, pkt_range)

    # 2. Add additive Gaussian noise to the IQ samples.
    data = awgn(data, snr_range)

    # 3. Convert time-domain IQ samples to channel-independent spectrograms.
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)

    # 4. Specify hyperparameters.
    margin = 0.5
    batch_size = 32

    # 5. Instantiate TripletNet (this is your feature extractor).
    triplet_net = TripletNet(data.shape, margin).to(device)

    # 6. If a pretrained model path is provided, load only the encoder weights.
    if pretrained_model_path is not None and os.path.isfile(pretrained_model_path):
        print(f"Loading pretrained encoder from: {pretrained_model_path}")
        # Make sure MyPretrainedModel has the same encoder architecture as TripletNet
        pretrained_model = MyPretrainedModel(input_shape=data.shape)
        pretrained_model.load_state_dict(
            torch.load(pretrained_model_path, map_location=device)
        )
        # Copy only the encoder weights
        triplet_net.encoder.load_state_dict(pretrained_model.encoder.state_dict())
        print("Pretrained encoder loaded successfully.")
    else:
        # If path provided but not found, just continue from scratch
        if pretrained_model_path is not None:
            print(f"Warning: {pretrained_model_path} not found. Training from scratch.")

    # 7. Split dataset into train/validation.
    data_train, data_valid, label_train, label_valid = train_test_split(
        data, label, test_size=0.1, shuffle=True
    )
    del data, label

    # 8. Create training and validation generators
    train_generator = triplet_net.create_generator(batch_size, dev_range, data_train, label_train)
    valid_generator = triplet_net.create_generator(batch_size, dev_range, data_valid, label_valid)

    # 9. Define optimizer and LR scheduler.
    optimizer = optim.Adam(triplet_net.parameters(), lr=2e-4)
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=10,  optimizer=optimizer)

    best_val_loss = float('inf')
    best_model = None

    # 10. Training loop

    num_classes = len(np.unique(label_train.cpu().numpy()))  # Get number of classes in train set
    maha_loss_fn = MahalanobisTripletLoss(margin=margin)        # Instantiate ONCE

    for epoch in range(num_epochs):
        triplet_net.train()
        train_loss = 0
        for step, (inputs, _) in enumerate(train_generator):
            anchor, positive, negative, labels = [inp.to(device) for inp in inputs]
            optimizer.zero_grad()

            embeddings = torch.cat([anchor, positive, negative], dim=0)
            labels_all = torch.cat([labels, labels, labels], dim=0)
            class_means, cov = estimate_class_means_and_global_cov(embeddings, labels_all, num_classes)
            cov_inv = torch.linalg.inv(cov)
            loss = maha_loss_fn(anchor, positive, negative, labels, class_means, cov_inv)


            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if step >= len(data_train) // batch_size:
                break
        train_loss /= (len(data_train) // batch_size)

        # ---- Validation steps ----
        triplet_net.eval()
        val_loss = 0
        with torch.no_grad():


            for step, (inputs, _) in enumerate(valid_generator):
                inputs = [inp.to(device) for inp in inputs]
                anchor, positive, negative, labels = inputs  # shape: (B, D), (B, D), (B, D), (B,)
                embeddings = torch.cat([anchor, positive, negative], dim=0)
                labels_all = torch.cat([labels, labels, labels], dim=0)
                class_means, cov = estimate_class_means_and_global_cov(embeddings, labels_all, num_classes)
                cov_inv = torch.linalg.inv(cov)
                loss = maha_loss_fn(anchor, positive, negative, labels, class_means, cov_inv)
                val_loss += loss.item()
                if step >= len(data_valid) // batch_size:
                    break
                    
        val_loss /= (len(data_valid) // batch_size)

        # Check if current validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = triplet_net
            os.makedirs(best_model_path, exist_ok=True)
            torch.save(
                best_model.state_dict(),
                os.path.join(best_model_path, 'best_model.pth')
            )
            print(f"Epoch {epoch + 1}: New best model saved with val_loss {best_val_loss:.8f}")

        # Adjust learning rate if needed
        last_lr = reduce_lr.get_last_lr()
        reduce_lr.step(val_loss)

        print(
            f"Epoch {epoch + 1}/500, "
            f"Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"learning rate: {last_lr}"
        )

    return best_model  # or return None if you prefer

def train_feature_extractor_hcf(
    file_path="/sise/home/talkoz/omer_model_based/LoRa_RFFI_dataset/dataset/Train/dataset_training_aug.h5",
    dev_range=np.arange(0, 30, dtype=int),
    pkt_range=np.arange(0, 1000, dtype=int),
    snr_range=np.arange(20, 80),
    best_model_path="/sise/home/talkoz/omer_model_based/new_stage_2/",
    pretrained_model_path="/sise/home/talkoz/omer_model_based/new_stage_1/best_classifier_hcf_84_07.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load IQ samples + labels
    raw_data, label = LoadDataset().load_iq_samples(file_path, dev_range, pkt_range)

    # 2. Add AWGN
    data_noisy = awgn(raw_data, snr_range)

    # 3. Hand-crafted RF features
    extra_features = []
    for sig in data_noisy:
        amp_imb, _, CFO, LO_leak = compute_rf_features(sig)
        extra_features.append([amp_imb, CFO, LO_leak])
    extra_features = torch.tensor(np.array(extra_features), dtype=torch.float32)

    # 4. Channel‑independent spectrograms
    spec_data = ChannelIndSpectrogram().channel_ind_spectrogram(data_noisy)
    spec_data = torch.tensor(spec_data, dtype=torch.float32).permute(0, 3, 1, 2)

    # 5. Hyper‑params
    margin     = 0.7
    batch_size = 32

    # 6. Model
    triplet_net = TripletNet_hcf(spec_data.shape, margin).to(device)

    # 7. Load pretrained encoder weights (if provided)
    if pretrained_model_path and os.path.isfile(pretrained_model_path):
        print(f'Loading pretrained weights: {pretrained_model_path}')
        from models import VQVAE_s_no_q_hcf  # adjust import if needed
        pre = VQVAE_s_no_q_hcf((1, 102, 62), 128, 30, 30, 30, 0.1)
        pre.load_state_dict(torch.load(pretrained_model_path, map_location=device))

        new_sd = {}
        for k, v in pre.encoder.state_dict().items():
            new_sd[k.replace('encoder.', '')] = v
        for k, v in pre.extra_fc.state_dict().items():
            new_sd[f'extra_fc.{k}'] = v
        for k, v in pre.concat_fc.state_dict().items():
            new_sd[f'concat_fc.{k}'] = v
        triplet_net.embedding_net.load_state_dict(new_sd, strict=False)
        print('Pretrained encoder loaded.')

    # 8. Train / validation split
    data_tr, data_val, extra_tr, extra_val, lbl_tr, lbl_val = train_test_split(
        spec_data, extra_features, label, test_size=0.1, shuffle=True
    )
    del raw_data, data_noisy, spec_data, extra_features, label

    # 9. Generators (assume you have create_generator)
    train_gen = triplet_net.create_generator(batch_size, dev_range, data_tr, extra_tr, lbl_tr)
    valid_gen = triplet_net.create_generator(batch_size, dev_range, data_val, extra_val, lbl_val)
    sil_loader = DataLoader(
        RFSingleDataset(data_val, extra_val, lbl_val),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )


    # 10. Optimiser / scheduler
    optimiser = optim.Adam(triplet_net.parameters(), lr=2e-4)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.2, patience=10)

    best_val_loss = float('inf')
    best_sil_score = float('inf')
    best_model_sd = None

    num_classes = len(np.unique(lbl_tr))  # Number of training classes
    maha_loss_fn = MahalanobisTripletLoss(margin=margin)  # Assuming you have this loss function

    #===================Diag===================================
    diag_cov = False
    #===================Diag===================================
    for epoch in range(500):
        # ---- train ----
        triplet_net.train()
        train_loss = 0
        for step, (inp, labs) in enumerate(train_gen):
            # inp = [spec, extra, ...], labs

            spec_a, extra_a, spec_p, extra_p, spec_n, extra_n = [t.to(device) for t in inp]
            anchor_labels, pos_labels, neg_labels = [l.to(device) for l in labs]         
            labels_all = torch.cat([anchor_labels, pos_labels, neg_labels], dim=0)
            if labels_all.ndim > 1:
                labels_all = labels_all.view(-1)  # flatten to [96]
            # Try to find out: what is the true label for the positives and negatives in this batch?
            # For example, if you have a way to retrieve the indices in the dataset for positive/negative,
            # print their labels as well.
            optimiser.zero_grad()
            
            # Get embeddings for anchor/pos/neg
            anchor_emb = triplet_net.encode(spec_a, extra_a)  # [B, D]
            pos_emb    = triplet_net.encode(spec_p, extra_p)
            neg_emb    = triplet_net.encode(spec_n, extra_n)

            
            # Stack all embeddings and labels for covariance calculation
            embeddings = torch.cat([anchor_emb, pos_emb, neg_emb], dim=0)
            
            # Estimate class means and global covariance for current batch
            class_means, cov = estimate_class_mean_cov(embeddings, labels_all, num_classes,diag_cov=diag_cov)
            cov_inv = torch.linalg.inv(cov)
            
            # Calculate Mahalanobis triplet loss
            loss = maha_loss_fn(anchor_emb, pos_emb, neg_emb, cov_inv)
            
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            if step >= len(data_tr) // batch_size:
                break
        train_loss /= (len(data_tr) // batch_size)

        # ---- validate ----
        triplet_net.eval()
        val_loss   = 0
        with torch.no_grad():
            for step, (inp, labs) in enumerate(valid_gen):
                spec_a, extra_a, spec_p, extra_p, spec_n, extra_n = [t.to(device) for t in inp]
                anchor_emb = triplet_net.encode(spec_a, extra_a)  # [B, D]
                pos_emb    = triplet_net.encode(spec_p, extra_p)
                neg_emb    = triplet_net.encode(spec_n, extra_n)
                anchor_labels, pos_labels, neg_labels = [l.to(device) for l in labs]  # each [32, 1]
                labels_all = torch.cat([anchor_labels, pos_labels, neg_labels], dim=0)
                if labels_all.ndim > 1:
                    labels_all = labels_all.view(-1)  # flatten to [96]
                embeddings = torch.cat([anchor_emb, pos_emb, neg_emb], dim=0)
                class_means, cov = estimate_class_mean_cov(embeddings, labels_all, num_classes,diag_cov=diag_cov)
                cov_inv = torch.linalg.inv(cov)
                loss = maha_loss_fn(anchor_emb, pos_emb, neg_emb, cov_inv)
                val_loss += loss.item()
                if step >= len(data_val) // batch_size:
                    break
        val_loss /= (len(data_val) // batch_size)
      

        print(
            f'Epoch {epoch+1:3d}/500 | '
            f'Train {train_loss:.8f} | Val {val_loss:.8f}'
        )
         # ---------------------- silhouette score ---------------
        latents = []
        labels  = []
        with torch.no_grad():
            for spec_batch, extra_batch, lbl_batch in sil_loader:
                spec_batch  = spec_batch.to(device)
                extra_batch = extra_batch.to(device)
                emb = triplet_net.encode(spec_batch, extra_batch)
                latents.append(emb.cpu())
                labels.append(lbl_batch)
        lat_all = torch.cat(latents, 0).numpy()
        lab_all = torch.cat(labels, 0).numpy().ravel()
        if np.unique(lab_all).size > 1:
            sil = silhouette_score(lat_all, lab_all)
        else:
            sil = float('nan')   # should not happen with stratified split
        print(f'Silhoutte score : {sil:.4f}') 
        # save best on val loss
        last_model = triplet_net.state_dict()
        os.makedirs(best_model_path, exist_ok=True)
        if diag_cov:
            torch.save(last_model, os.path.join(best_model_path, 'last_hcf_model_diag_cov.pth'))
        else:
            torch.save(last_model, os.path.join(best_model_path, 'last_hcf_model.pth'))
        if val_loss < best_val_loss or (val_loss == best_val_loss and sil > best_sil_score):
            best_sil_score = sil
            best_val_loss = val_loss
            best_model_sd = triplet_net.state_dict()
            os.makedirs(best_model_path, exist_ok=True)
            if diag_cov:
                torch.save(last_model, os.path.join(best_model_path, 'last_hcf_model_diag_cov.pth'))
            else:
                torch.save(best_model_sd, os.path.join(best_model_path, 'best_hcf_model.pth'))
            print(f'  ↳  saved new best model (val_loss {best_val_loss:.4f})')

        last_lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

    # return best model
    if best_model_sd is not None:
        triplet_net.load_state_dict(best_model_sd)
    return triplet_net

    

def test_classification(
        file_path_enrol,
        file_path_clf,
        feature_extractor_name,
        dev_range_enrol=np.arange(30, 40, dtype=int),
        pkt_range_enrol=np.arange(0, 100, dtype=int),
        dev_range_clf=np.arange(30, 40, dtype=int),
        pkt_range_clf=np.arange(100, 200, dtype=int)
):
    '''
    test_classification performs a classification task and returns the 
    classification accuracy.
    
    INPUT: 
        FILE_PATH_ENROL is the path of enrollment dataset.
        
        FILE_PATH_CLF is the path of classification dataset.
        
        FEATURE_EXTRACTOR_NAME is the name of RFF extractor used during 
        enrollment and classification. 
        
        DEV_RANGE_ENROL is the label range of LoRa devices during enrollment.
        
        PKT_RANGE_ENROL is the range of packets from each LoRa device during enrollment.
        
        DEV_RANGE_CLF is the label range of LoRa devices during classification.
        
        PKT_RANGE_CLF is the range of packets from each LoRa device during classification.

    RETURN:
        PRED_LABEL is the list of predicted labels.
        
        TRUE_LABEL is the list true labels.
        
        ACC is the overall classification accuracy.
    '''

    # Load the saved RFF extractor.
    feature_extractor = load_model(feature_extractor_name, compile=False)

    LoadDatasetObj = LoadDataset()

    # Load the enrollment dataset. (IQ samples and labels)
    data_enrol, label_enrol = LoadDatasetObj.load_iq_samples(file_path_enrol,
                                                             dev_range_enrol,
                                                             pkt_range_enrol)

    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    # Convert IQ samples to channel independent spectrograms. (enrollment data)
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol)

    # # Visualize channel independent spectrogram
    # plt.figure()
    # sns.heatmap(data_enrol[0,:,:,0],xticklabels=[], yticklabels=[], cmap='Blues', cbar=False)
    # plt.gca().invert_yaxis()
    # plt.savefig('channel_ind_spectrogram.pdf')

    # Extract RFFs from channel independent spectrograms.
    feature_enrol = feature_extractor.predict(data_enrol)
    del data_enrol

    # Create a K-NN classifier using the RFFs extracted from the enrollment dataset.
    knnclf = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
    knnclf.fit(feature_enrol, np.ravel(label_enrol))

    # Load the classification dataset. (IQ samples and labels)
    data_clf, true_label = LoadDatasetObj.load_iq_samples(file_path_clf,
                                                          dev_range_clf,
                                                          pkt_range_clf)

    # Convert IQ samples to channel independent spectrograms. (classification data)
    data_clf = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_clf)

    # Extract RFFs from channel independent spectrograms.
    feature_clf = feature_extractor.predict(data_clf)
    del data_clf

    # Make prediction using the K-NN classifier.
    pred_label = knnclf.predict(feature_clf)

    # Calculate classification accuracy.
    acc = accuracy_score(true_label, pred_label)
    print('Overall accuracy = %.4f' % acc)

    return pred_label, true_label, acc


def test_rogue_device_detection(
        feature_extractor_name,
        file_path_enrol='C:/Users/omer1/Desktop/studies/thesis/LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5',
        dev_range_enrol=np.arange(30, 40, dtype=int),
        pkt_range_enrol=np.arange(0, 100, dtype=int),
        file_path_legitimate='C:/Users/omer1/Desktop/studies/thesis/LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5',
        dev_range_legitimate=np.arange(30, 40, dtype=int),
        pkt_range_legitimate=np.arange(100, 200, dtype=int),
        file_path_rogue='C:/Users/omer1/Desktop/studies/thesis/LoRa_RFFI_dataset/dataset/Test/dataset_rogue.h5',
        dev_range_rogue=np.arange(40, 45, dtype=int),
        pkt_range_rogue=np.arange(0, 100, dtype=int),
        max_clusters=20,
):
    '''
    test_rogue_device_detection performs the rogue device detection task using
    a specific RFF extractor. It returns false positive rate (FPR), true 
    positive rate (TPR), area under the curve (AUC) and corresponding threshold 
    settings.
    
    INPUT: 
    
        FEATURE_EXTRACTOR_NAME is the name of RFF extractor used in rogue 
        device detection.
        
        FILE_PATH_ENROL is the path of enrollment dataset.
        
        DEV_RANGE_ENROL is the device index range used in the enrollment stage.
        
        PKT_RANGE_ENROL is the packet index range used in the enrollment stage.
        
        FILE_PATH_LEGITIMATE is the path of dataset contains packets from
        legitimate devices.
        
        DEV_RANGE_LEGITIMATE is the index range of legitimate devices used in
        the rogue device detection stage.
        
        PKT_RANGE_LEGITIMATE specifies the packet range from legitimate devices 
        used in the rogue device detection stage.
        
        FILE_PATH_ROGUE is the path of dataset contains packets from rogue 
        devices.
        
        DEV_RANGE_ROGUE is the index range of rogue devices used in the rogue 
        device detection stage.
        
        PKT_RANGE_ROGUE specifies the packet range from rogue devices used in 
        the rogue device detection stage.
    
    RETURN:
        FPR is the detection false positive rate.
        
        TRP is the detection true positive rate.

        ROC_AUC is the area under the ROC curve.
        
        EER is the equal error rate.
        
    '''

    def _compute_eer(fpr, tpr, thresholds):
        '''
        _COMPUTE_EER returns equal error rate (EER) and the threshold to reach
        EER point.
        '''
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))

        return eer, thresholds[min_index]

    # Load RFF extractor.
    feature_extractor = load_model(feature_extractor_name, compile=False)

    LoadDatasetObj = LoadDataset()

    # Load enrollment dataset.
    data_enrol, label_enrol = LoadDatasetObj.load_iq_samples(file_path_enrol,
                                                             dev_range_enrol,
                                                             pkt_range_enrol)

    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    # Convert IQ samples to channel independent spectrograms.
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol)

    # Extract RFFs from cahnnel independent spectrograms.
    feature_enrol = feature_extractor.predict(data_enrol)
    del data_enrol
    # feature_representation.visualize_feature_space(feature_enrol, label_enrol, 'pca')
    # Build a K-NN classifier.
    knnclf = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
    knnclf.fit(feature_enrol, np.ravel(label_enrol))

    # Load the test dataset of legitimate devices.
    data_legitimate, label_legitimate = LoadDatasetObj.load_iq_samples(file_path_legitimate,
                                                                       dev_range_legitimate,
                                                                       pkt_range_legitimate)
    # Load the test dataset of rogue devices.
    data_rogue, label_rogue = LoadDatasetObj.load_iq_samples(file_path_rogue,
                                                             dev_range_rogue,
                                                             pkt_range_rogue)

    # Combine the above two datasets into one dataset containing both rogue
    # and legitimate devices.
    data_test = np.concatenate([data_legitimate, data_rogue])
    label_test = np.concatenate([label_legitimate, label_rogue])
    label_test = label_test.reshape(1500)
    label_enrol = label_enrol.reshape(1000)
    label_legitimate = label_legitimate.reshape(1000)
    label_rogue = label_rogue.reshape(500)
    # Convert IQ samples to channel independent spectrograms.
    data_test = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_test)

    # Extract RFFs from channel independent spectrograms.
    feature_test = feature_extractor.predict(data_test)
    del data_test
    # feature_representation.visualize_feature_space(feature_test, label_test)
    # Find the nearest 15 neighbors in the RFF database and calculate the 
    # distances to them.
    distances, indexes = knnclf.kneighbors(feature_test)

    # Calculate the average distance to the nearest 15 neighbors.
    detection_score = distances.mean(axis=1)

    # Label the packets sent from legitimate devices as 1. The rest are sent by rogue devices
    # and are labeled as 0.
    true_label = np.zeros([len(label_test), 1])
    true_label[(label_test <= dev_range_legitimate[-1]) & (label_test >= dev_range_legitimate[0])] = 1

    # Compute receiver operating characteristic (ROC).
    fpr, tpr, thresholds = roc_curve(true_label, detection_score, pos_label=1)

    # The Euc. distance is used as the detection score. The lower the value, 
    # the more similar it is. This is opposite with the probability or confidence 
    # value used in scikit-learn roc_curve function. Therefore, we need to subtract 
    # them from 1.
    fpr = 1 - fpr
    tpr = 1 - tpr

    # Compute EER.
    eer, _ = _compute_eer(fpr, tpr, thresholds)
    predicted_labels_knn = [1 if detection_score[j] <= 0.39776 else 0 for j in range(len(detection_score))]
    pfa_knn, pmd_knn, acc_knn = calculate_metrics(true_label, predicted_labels_knn)
    # Compute AUC.
    roc_auc = auc(fpr, tpr)
    all_features = np.concatenate([feature_enrol, feature_test])
    all_labels = np.concatenate([label_enrol, label_test])

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    features_tsne = tsne.fit_transform(all_features)

    inertia_values = []  # To store inertia values for the Elbow Method
    best_k = None
    best_score = float('inf')
    matched_centroids = None
    k_range = range(10, max_clusters + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k).fit(all_features.astype(np.float64))
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_[:len(feature_enrol)]  # Only use labels for feature_enrol

        # Store the inertia for Elbow Method
        inertia_values.append(kmeans.inertia_)

        # Create a cost matrix based on label_enrol only
        cost_matrix = np.zeros((k, len(dev_range_enrol)))

        for i in range(k):  # For each cluster
            for j, label in enumerate(dev_range_enrol):  # For each device in dev_range_enrol (30 to 39)
                # Calculate mismatch count for cluster i to match label
                cost_matrix[i, j] = np.sum((labels == i) != (label_enrol == label))

        # Solve the assignment problem using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matching_score = cost_matrix[row_ind, col_ind].sum()

        if matching_score < best_score:
            best_score = matching_score
            best_k = k
            matched_centroids = (row_ind, col_ind)  # Save the optimal assignment
            legitimate_centroids = set(row_ind)  # Track legitimate clusters
            best_kmeans = kmeans

    # Plot the Elbow Method graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia_values, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig('elbow_method_kmeans.png')
    plt.close()


    # Transform feature_test using the same t-SNE
    feature_test_tsne = tsne.fit_transform(feature_test)
    test_labels = best_kmeans.labels_[1000:]

    # Assign clusters as legitimate or rogue using the optimal assignment
    row_ind, col_ind = matched_centroids

    # Map predicted test labels to legitimate or rogue
    predicted_labels = np.array([1 if label in legitimate_centroids else 0 for label in test_labels])
    true_labels = np.array([1 if label in dev_range_legitimate else 0 for label in label_test])

    # Calculate ROC metrics
    fpr_kmeans, tpr_kmeans, thresholds_kmeans = roc_curve(true_labels, predicted_labels, pos_label=1)
    roc_auc_kmeans = auc(fpr_kmeans, tpr_kmeans)
    eer_kmeans, _ = _compute_eer(fpr_kmeans, tpr_kmeans, thresholds_kmeans)

    # Visualize the decision boundaries
    # feature_representation.visualize_legitimate_rogue_with_decision_boundary(
    #     all_features,
    #     true_labels,
    #     best_kmeans.cluster_centers_,
    #     best_kmeans,
    #     legitimate_centroids,
    #     title="Decision Boundaries for Legitimate and Rogue Devices"
    # )
    pfa_kmeans, pmd_kmeans, acc_kmeans = calculate_metrics(predicted_labels, true_labels)
    print(f'kmeans P_FA = {pfa_kmeans}\n kmeans P_MD = {pmd_kmeans}\n kmeans accuracy = {acc_kmeans}'
          f'\n KNN P_FA = {pfa_knn}\n KNN P_MD = {pmd_knn}\n KNN accuracy = {acc_knn}')
    return fpr, tpr, roc_auc, eer, best_k, fpr_kmeans, tpr_kmeans, roc_auc_kmeans, eer_kmeans


if __name__ == '__main__':

    # Specifies what task the program runs for. 
    # 'Train'/'Classification'/'Rogue Device Detection'
    run_for = 'Train_hcf'

    if run_for == 'Train':

        # Train an RFF extractor.
        feature_extractor = train_feature_extractor()
        # Save the trained model.
        feature_extractor.save('Extractor.keras')
    if run_for == 'Train_hcf':

        # Train an RFF extractor.
        feature_extractor = train_feature_extractor_hcf()
        # Save the trained model.
        feature_extractor.save('Extractor.keras')


    elif run_for == 'Classification':

        # Specify the device index range for classification.
        test_dev_range = np.arange(30, 40, dtype=int)

        # Perform the classification task.
        pred_label, true_label, acc = test_classification(file_path_enrol=
                                                          "/sise/home/talkoz/omer_model_based/LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5",
                                                          file_path_clf=
                                                          "/sise/home/talkoz/omer_model_based/LoRa_RFFI_dataset/dataset/Test/channel_problem/A.h5",
                                                          feature_extractor_name=
                                                          "/sise/home/talkoz/omer_model_based/LoRa_RFFI_dataset/models/Extractor_1.h5")

        # Plot the confusion matrix.
        conf_mat = confusion_matrix(true_label, pred_label)
        classes = test_dev_range + 1

#        plt.figure()
#        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
#                    xticklabels=classes, yticklabels=classes)
#        plt.xlabel('Predicted label', fontsize=20)
#        plt.ylabel('True label', fontsize=20)
#        plt.savefig('confusion_matrix.png')
#        plt.close()
#        



    elif run_for == 'Rogue Device Detection':

        # Perform rogue device detection task using three RFF extractors.
        fpr, tpr, roc_auc, eer, est_k, fpr_kmeans, tpr_kmeans, roc_auc_kmeans, eer_kmeans = test_rogue_device_detection(
            "C:/Users/omer1/Desktop/studies/thesis/lora_liverpool/LoRa_RFFI-main/Openset_RFFI_TIFS/Extractor.h5")

        # Plot the ROC curves.
        plt.figure(figsize=(4.8, 2.8))
        plt.xlim(-0.01, 1.02)
        plt.ylim(-0.01, 1.02)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Extractor 1, AUC = ' +
                                 str(round(roc_auc, 3)) + ', EER = ' + str(round(eer, 3)))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc=4)
        plt.savefig('roc_curve.png')
        plt.close()
        
        plt.figure(figsize=(4.8, 2.8))
        plt.xlim(-0.01, 1.02)
        plt.ylim(-0.01, 1.02)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_kmeans, tpr_kmeans, label='Extractor 1, AUC = ' +
                                 str(round(roc_auc_kmeans, 3)) + ', EER = ' + str(round(eer_kmeans, 3)))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC curve kmeans K = {est_k}')
        plt.legend(loc=4)
        plt.savefig("roc_curve_kmeans.png")
        plt.close()
        

