###################################################################################################
###################################################################################################
#################################        Adaptive Codebook        #################################
################################################################################################
###################################################################################################

''' Here the implementation of Learning Multi-Rate Vector Quantization for Remote Deep Inference by
May Malka, Shai Ginzach, and Nir Shlezinger

For further questions: maymal@post.bgu.ac.il
'''

###################################################################################################
###################################################################################################
#################################             Imports             #################################
###################################################################################################
###################################################################################################

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import time
from sklearn.metrics import confusion_matrix, silhouette_score

import os
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import math
import wandb

from models import VQVAE, VQVAE_s, VQVAE_no_q, VQVAE_s_no_q, VQVAE_s_no_q_hcf,VQVAE_s_hcf
from hand_crafted_features import compute_rf_features

from liverpool.closeset.dataset_preparation import LoadDataset, awgn, ChannelIndSpectrogram
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


###################################################################################################
###################################################################################################
##################################       Training Function        #################################
###################################################################################################
###################################################################################################


def train_fr(model, optimizer, criterion, trainloader, valloader, device, epochs=500, save_path="best_model.pth"):
    wandb.watch(model, log="all")  # Log gradients and parameters
    base_path = "plots"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(base_path, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    start_time = time.time()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_num, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            if labels.ndim > 1:  # If labels are one-hot encoded, convert to class indices
                labels = labels.flatten()

            # Forward pass
            preds, quant_loss = model(inputs)

            # Compute loss
            ce_loss = criterion(preds, labels)
            if quant_loss:
                loss = ce_loss + quant_loss
            else:
                loss = ce_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct_train += model.get_accuracy(labels, preds)
            total_train += labels.size(0)

        train_loss = epoch_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc})
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.2f}%, LR: {scheduler.get_last_lr()}")

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                if labels.ndim > 1:
                    labels = labels.flatten()

                preds, quant_loss = model(inputs)
                ce_loss = criterion(preds, labels)
                if quant_loss:
                    loss = ce_loss + quant_loss
                else:
                    loss = ce_loss

                val_loss += loss.item()
                correct_val += model.get_accuracy(labels, preds)
                total_val += labels.size(0)

                predicted_labels = torch.argmax(preds, dim=1)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predicted_labels.cpu().numpy())

            val_loss /= len(valloader)
            val_acc = 100 * correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch [{epoch + 1}], Val Loss: {val_loss:.6f}, Val Accuracy: {val_acc:.2f}%")

        # --- Save best model & compute metrics if improved ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model at Epoch {epoch + 1} with Val Accuracy: {val_acc:.2f}%")

            # Compute confusion matrix & accuracy per class
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            classes = np.unique(all_labels)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            cm_filename = os.path.join(run_dir, f"confusion_matrix_{timestamp}.png")
            plt.savefig(cm_filename)
            plt.close()
            print(f"Saved confusion matrix to {cm_filename}")

            # Per-class accuracy
            acc_per_class = {}
            for i, cls in enumerate(classes):
                total = np.sum(cm[i, :])
                acc = 100 * cm[i, i] / total if total > 0 else 0
                acc_per_class[cls] = acc

            # Bar chart: per-class accuracy
            plt.figure(figsize=(10, 6))
            plt.bar(list(acc_per_class.keys()), list(acc_per_class.values()), color='skyblue')
            plt.xlabel("Device/Class")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy per Device")
            plt.ylim([0, 100])
            for cls, acc in acc_per_class.items():
                plt.text(cls, acc + 1, f"{acc:.2f}%", ha='center')
            plt.tight_layout()
            bar_filename = os.path.join(run_dir, f"accuracy_per_device_{timestamp}.png")
            plt.savefig(bar_filename)
            plt.close()
            print(f"Saved accuracy per device bar diagram to {bar_filename}")

            # -------------- NEW CODE FOR CODEBOOK USAGE --------------
            # If the model uses a quantizer, track the usage of each code index.
            # We'll do another pass over val_loader to see how often each codeword is used.
            if hasattr(model, 'quantizer'):
                # Initialize a counter for each codeword
                num_codewords = model.quantizer.num_embeddings  # or your codebook dimension
                codeword_counts = np.zeros(num_codewords, dtype=int)

                # Collect code indices on the validation set
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs = inputs.to(device)
                        # In your model's forward(), ensure you can retrieve the code indices.
                        # For example, you might have something like:
                        #   preds, quant_loss, code_indices = model(inputs, return_indices=True)
                        # or store code_indices as an attribute inside the model.quantizer.
                        
                        # If your forward call doesn't return indices, you'll need to modify
                        # the model/quantizer to give them back. Below is an example pattern:
                        preds, quant_loss, code_indices = model(inputs, return_indices=True)

                        # code_indices should be a tensor of shape [batch_size] or [batch_size, ...]
                        # Flatten if needed:
                        code_indices = code_indices.view(-1).cpu().numpy()
                        for idx in code_indices:
                            codeword_counts[idx] += 1

                # Plot usage as a bar chart
                plt.figure(figsize=(10, 6))
                plt.bar(range(num_codewords), codeword_counts, color='tab:purple')
                plt.xlabel("Codeword Index")
                plt.ylabel("Usage Count")
                plt.title("Codeword Usage Frequency")
                usage_path = os.path.join(run_dir, f"codeword_usage_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(usage_path)
                plt.close()
                print(f"Saved codeword usage bar chart to {usage_path}")
            # -------------- END NEW CODE FOR CODEBOOK USAGE --------------

        # Step the scheduler
        scheduler.step(val_loss)
    
    duration = time.time() - start_time
    print(f"Training completed in {duration / 3600:.2f} hours")
    return train_losses, train_accuracies, val_losses, val_accuracies


def train_fr_hcf(model, optimizer, criterion, trainloader, valloader, device, 
                 epochs=500, save_path="best_model.pth"):
    """
    Train loop for an HCF model. Now tracks silhouette score of latent space
    (z_q) each epoch in the validation phase.
    """
    wandb.watch(model, log="all")  # Log gradients and parameters
    base_path = "plots"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(base_path, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=30)
    # # Example scheduler
    # scheduler = CyclicLR(
    #     optimizer,
    #     base_lr=5e-7,
    #     max_lr=5e-5,
    #     step_size_up=20,
    #     mode='triangular',
    #     cycle_momentum=False  # turn off momentum tweaks if using Adam
    # )
   # scheduler = CosineAnnealingLR(
   #     optimizer,
   #     T_max=350  # Number of epochs you want to train for
   #     # (Optionally: eta_min=1e-6)  # Minimum learning rate
   # )
    current_val_loss = 999999999999
    for epoch in range(epochs):
        # -------------------- Training --------------------
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_num, (inputs, extra_features, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.flatten()

            # Forward pass
            output = model(inputs, extra_features, return_latents=False)
            # output could be (preds, quant_loss) or (preds, quant_loss, code_indices)
            if len(output) == 2:
                preds, quant_loss = output
            else:
                preds, quant_loss, code_indices = output

            ce_loss = criterion(preds, labels)
            if quant_loss is not None:
                loss = ce_loss + quant_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct_train += model.get_accuracy(labels, preds)
            total_train += labels.size(0)

        train_loss = epoch_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc})
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, "
              f"Train Accuracy: {train_acc:.2f}%, LR: {scheduler.get_last_lr()[0]}")

        # -------------------- Validation --------------------
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # We'll collect latents here to compute silhouette after the loop
        all_labels = []
        all_preds = []
        all_latents = []

        with torch.no_grad():
            for inputs, extra_features, labels in valloader:
                inputs = inputs.to(device)
                extra_features = extra_features.to(device)
                labels = labels.to(device)
                if labels.ndim > 1:
                    labels = labels.flatten()

                # Forward pass with return_latents=True
                output = model(inputs, extra_features, return_latents=True)
                # Now output is (preds, quant_loss, code_indices, z_q)
                if len(output) == 4:
                    preds, quant_loss, code_indices, z_e = output
                else:
                    # fallback if older model version
                    preds, quant_loss = output
                    z_e = None
                    print("Z_E is none")

                ce_loss = criterion(preds, labels)
                if quant_loss is not None:
                    loss = ce_loss + quant_loss
                else:
                    loss = ce_loss

                val_loss += loss.item()
                correct_val += model.get_accuracy(labels, preds)
                total_val += labels.size(0)

                predicted_labels = torch.argmax(preds, dim=1)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predicted_labels.cpu().numpy())

                # Collect latents for silhouette (only if we have z_q)
                if z_e is not None:
                    # Move to CPU numpy for silhouette
                    all_latents.append(z_e.detach().cpu().numpy())

            val_loss /= len(valloader)
            val_acc = 100 * correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch [{epoch + 1}/{epochs}], Val Loss: {val_loss:.6f}, "
                  f"Val Accuracy: {val_acc:.2f}%")

            # ---------- Compute Silhouette Score on z_q ----------
            # Make sure there is more than one class in your batch or silhouette_score will fail
            if len(all_latents) > 0: 
                latents_concat = np.concatenate(all_latents, axis=0)
                labels_concat = np.concatenate(all_labels, axis=0)
                preds_concat = np.concatenate(all_preds, axis=0)
                unique_labels = np.unique(labels_concat)
                if len(unique_labels) > 1:
#                    sil_score_true = silhouette_score(latents_concat, labels_concat)
#                    wandb.log({"epoch": epoch + 1, "silhouette_score_true": sil_score_true})
#                    print(f"Silhouette Score (z_e vs. true labels): {sil_score_true:.4f}")
#                    sil_score_pred = silhouette_score(latents_concat, preds_concat)
#                    wandb.log({"epoch": epoch + 1, "silhouette_score_pred": sil_score_pred})
#                    print(f"Silhouette Score (z_e vs. pred labels): {sil_score_pred:.4f}")
                    unique_labels = np.unique(labels_concat)
                    if len(unique_labels) > 1:
                        sil_score_true = silhouette_score(latents_concat, labels_concat)
                        print(f"Silhouette Score (z_e vs. true labels): {sil_score_true:.4f}")
                    else:
                        print("Silhouette score (true labels) requires at least 2 distinct labels, skipping.")
                    
                    unique_pred_labels = np.unique(preds_concat)
                    if len(unique_pred_labels) > 1:
                        sil_score_pred = silhouette_score(latents_concat, preds_concat)
                        print(f"Silhouette Score (z_e vs. pred labels): {sil_score_pred:.4f}")
                    else:
                        print("Silhouette score (pred labels) requires at least 2 distinct predicted labels, skipping.")

                else:
                    print("Silhouette score requires at least 2 distinct labels, skipping.")

        # -------------------- Check if best model --------------------
        if val_acc > best_val_acc or (val_acc==best_val_acc and current_val_loss > val_loss) :
            current_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model at Epoch {epoch + 1} with Val Accuracy: {val_acc:.2f}%")

            # Build confusion matrix on the entire val set
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            classes = np.unique(all_labels)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            cm_filename = os.path.join(run_dir, f"confusion_matrix_{timestamp}.png")
            plt.savefig(cm_filename)
            plt.close()
            print(f"Saved confusion matrix to {cm_filename}")

            # Accuracy per class bar chart
            acc_per_class = {}
            for i, cls in enumerate(classes):
                total_cls = np.sum(cm[i, :])
                acc_cls = 100 * cm[i, i] / total_cls if total_cls > 0 else 0
                acc_per_class[cls] = acc_cls

            plt.figure(figsize=(10, 6))
            plt.bar(list(acc_per_class.keys()), list(acc_per_class.values()))
            plt.xlabel("Device/Class")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy per Device")
            plt.ylim([0, 100])
            for cls, acc_c in acc_per_class.items():
                plt.text(cls, acc_c + 1, f"{acc_c:.2f}%", ha='center')
            plt.tight_layout()
            bar_filename = os.path.join(run_dir, f"accuracy_per_device_{timestamp}.png")
            plt.savefig(bar_filename)
            plt.close()
            print(f"Saved accuracy per device bar chart to {bar_filename}")

            # ---- If model has a quantizer, plot code usage ----
            if hasattr(model, 'quantizer'):
                num_codewords = model.quantizer.p  # codebook_size
                codeword_counts = np.zeros(num_codewords, dtype=int)

                with torch.no_grad():
                    for inputs, extra_features, labels in valloader:
                        inputs = inputs.to(device)
                        extra_features = extra_features.to(device)
                        out = model(inputs, extra_features, return_latents=False)
                        if len(out) == 3:
                            _, _, code_indices = out
                            code_indices = code_indices.view(-1).cpu().numpy()
                            for idx in code_indices:
                                codeword_counts[idx] += 1

                # Plot code usage
                plt.figure(figsize=(10, 6))
                plt.bar(range(num_codewords), codeword_counts)
                plt.xlabel("Codeword Index")
                plt.ylabel("Usage Count")
                plt.title("Codeword Usage Frequency")
                code_usage_file = os.path.join(run_dir, f"codeword_usage_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(code_usage_file)
                plt.close()
                print(f"Saved codeword usage bar chart to {code_usage_file}")

        # Step the scheduler
        scheduler.step(val_loss)

    duration = time.time() - start_time
    print(f"Training completed in {duration / 3600:.2f} hours")
    return train_losses, train_accuracies, val_losses, val_accuracies



#################################################################################################
#################################################################################################
##############################      Train Adaptive Codebook       ###############################
#################################################################################################
#################################################################################################

def setup_config():
    """
    Define training parameters and mode switches.
    Adjust these flags and paths to select your desired training permutation.
    """
    config = {
        # Pretrained component flags:
        'pretrained_encoder': True,     # Load pretrained encoder weights?
        'pretrained_classifier': False,  # Load pretrained classifier weights?
        #WHEN STAGE 5 HANGE TO TRUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUE
        'pretrained_codebook': True,    # Load pretrained codebook weights? (Only used if use_quantizer is True)
        'load_codebook_from_model_weights': False,
        
        # Freeze options (if using pretrained parts):
        'freeze_encoder': True,          # Freeze encoder parameters after loading pretrained weights?
        'freeze_classifier': False,      # Freeze classifier parameters after loading pretrained weights?
        'freeze_codebook': False,        # Freeze codebook parameters after loading pretrained weights?
        'freeze_hcf_params': True,
        
        # Architecture options:
        'use_hcf': True,                 # Include hand-crafted features (HCF)?
        #WHEN STAGE 5 HANGE TO TRUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUE
        'use_quantizer': True,          # Use a model variant with a quantizer/codebook? #WHEN STAGE 5 HANGE TO TRUUUUUUUUUUUUUUUUUUUUUUUUE
        'diag_cov': True,
        'dropout' : False,
        'entropy_lambda' : 1,  # <-- Tune this! Start with 1e-2 or 1e-3

        
        'model_mode': 'close_set_enc',         # Options: 'close_set_enc', 'triplet', 'ss'
        'add_noise':False,
        'snr_range': range(20, 80),       # SNR range for AWGN
        'dev_range': range(30, 40),
        'pkt_range': range(0, 100),
        'path_to_lora_train': "/sise/home/talkoz/omer_model_based/LoRa_RFFI_dataset/dataset/Test/dataset_residential.h5",
        'input_shape': (1, 102, 62),      # Example spectrogram input shape: (channels, height, width)
        'num_hcf_features': 30,
        'num_classes': 10,
        'codebook_size': 128,
        'batch_size': 32,
        'epochs': 1000,
        'num_embeddings': 128,
        'seed':42, #was 1111
        
        # Pretrained weight file paths for model_mode 'close_set_enc':
        'encoder_weights_path_close_set_enc': "",
        'encoder_weights_path_hcf_close_set_enc': "/sise/home/talkoz/omer_model_based/new_stage_5/best_classifier_hcf_67.pth",
        'classifier_weights_path_close_set_enc': "",
        'classifier_weights_path_hcf_close_set_enc': "/sise/home/talkoz/omer_model_based/new_stage_5/best_classifier_hcf_67.pth",
        'codebook_weights_path_close_set_enc': "",
        'codebook_weights_path_hcf_close_set_enc': "/sise/home/talkoz/omer_model_based/new_stage_4/lbg_codebooks/vq_codebook_five_stage_size_diag_30_40.pth", #put the stage 4 weights!!!

        # Pretrained weight file paths for model_mode 'triplet':
        'encoder_weights_path_triplet': "vq_vae_models/encoder_classifier/encoder_weights_triplet.pth",
        'encoder_weights_path_hcf_triplet': "vq_vae_models/triplet_net/best_hcf_model.pth",
        'classifier_weights_path_triplet': "vq_vae_models/encoder_classifier/classifier_weights_triplet.pth",
        'classifier_weights_path_hcf_triplet': "vq_vae_models/encoder_classifier/hcf/best_classifier_79_enc_froz_triplet.pth",
        'codebook_weights_path_triplet': "vq_vae_models/codebooks/kmeans_codebooks/vq_codebook_triplet.pth",
        'codebook_weights_path_hcf_triplet': "vq_vae_models/codebooks/kmeans_codebooks/vq_codebook_hcf_triplet.pth",

        # Pretrained weight file paths for model_mode 'ss' (self-supervised):
        'encoder_weights_path_ss': "vq_vae_models/self_supervised_net_awgn/encoder_weights_ss.pth",
        'encoder_weights_path_hcf_ss': "vq_vae_models/self_supervised_net_awgn/hcf/encoder_weights_ss.pth",
        'classifier_weights_path_ss': "vq_vae_models/self_supervised_net_awgn/classifier_weights_ss.pth",
        'classifier_weights_path_hcf_ss': "vq_vae_models/self_supervised_net_awgn/hcf/classifier_weights_ss.pth",
        'codebook_weights_path_ss': "vq_vae_models/codebooks/kmeans_codebooks/vq_codebook_ss.pth",
        'codebook_weights_path_hcf_ss': "vq_vae_models/codebooks/kmeans_codebooks/vq_codebook_hcf_ss.pth",
    }
    # Set the save path for the trained model.
    if config['use_quantizer']:
        if config['model_mode'] == 'close_set_enc':
            if config['use_hcf']:
                config['save_path'] = "/sise/home/talkoz/omer_model_based/new_stage_5/best_classifier_hcf.pth"
            else:
                config['save_path'] = "vq_vae_models/vq_vae_classifier/close_set_enc/best_classifier.pth"
        elif config['model_mode'] == 'triplet':
            if config['use_hcf']:
                config['save_path'] = "vq_vae_models/vq_vae_classifier/triplet_enc/best_classifier_hcf.pth"
            else:
                config['save_path'] = "vq_vae_models/vq_vae_classifier/triplet_enc/best_classifier.pth"
    else:
        if config['model_mode'] == 'close_set_enc':
            if config['use_hcf']:
                config['save_path'] = "/sise/home/talkoz/omer_model_based/new_stage_3/best_classifier_hcf.pth"
            else:
                config['save_path'] = "vq_vae_models/encoder_classifier/close_set_enc/best_classifier.pth"
        elif config['model_mode'] == 'triplet':
            if config['use_hcf']:
                config['save_path'] = "vq_vae_models/encoder_classifier/triplet_enc/best_classifier_hcf.pth"
            else:
                config['save_path'] = "vq_vae_models/encoder_classifier/triplet_enc/best_classifier.pth"
    return config

def setup_data(config):
    # 1. Load IQ samples and labels.
    LoadDatasetObj = LoadDataset()
    data_train, label_train = LoadDatasetObj.load_iq_samples(
        file_path=config['path_to_lora_train'],
        dev_range=config['dev_range'],
        pkt_range=config['pkt_range']
    )

    # 2. Shuffle and adjust labels.
    index = np.arange(len(label_train))
    np.random.shuffle(index)
    data_train = data_train[index, :]
    label_train = label_train[index]
    label_train = label_train - config['dev_range'][0]
    label_train = torch.tensor(label_train, dtype=torch.long)

    # 3. Add AWGN.
    if config['add_noise']:
        data_train = awgn(data_train, config['snr_range'])

    # 4. Convert IQ samples to channel-independent spectrograms.
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    spectrogram_data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_train)
    spectrogram_data = torch.tensor(spectrogram_data, dtype=torch.float32).permute(0, 3, 1, 2)

    # 5. Create dataset.
    if config['use_hcf']:
        extra_features_list = []
        for signal in data_train:
            # compute_rf_features returns (amp imbalance, phase imbalance, CFO, LO leakage)
            amp_imbalance, _, CFO, LO_leakage = compute_rf_features(signal)
            extra_features_list.append([amp_imbalance, CFO, LO_leakage])
        extra_features = np.array(extra_features_list)
        extra_features = torch.tensor(extra_features, dtype=torch.float32)
        dataset = TensorDataset(spectrogram_data, extra_features, label_train)
    else:
        dataset = TensorDataset(spectrogram_data, label_train)

    # 6. Split dataset into training and validation sets.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_loader, val_loader

def build_model(config, device, dropout=False):
    """
    Build the appropriate model based on the configuration flags.
    """
    if config['use_quantizer']:
        # Models with quantizer.
        if config['use_hcf']:
            model = VQVAE_s_hcf(
                config['input_shape'],
                num_embeddings=config['num_embeddings'],
                hcf_n = config['num_hcf_features'],
                codebook_size=config['codebook_size'],
                num_classes=config['num_classes'],
                commitment_loss_weight = 0.1,
                diag_cov=config['diag_cov'],
                dropout = config['dropout']
            )
        else:
            model = VQVAE_s(
                config['input_shape'],
                num_embeddings=config['num_embeddings'],
                codebook_size=config['codebook_size'],
                num_classes=config['num_classes']
            )
    else:
        # Models without quantizer.
        if config['use_hcf']:
            model = VQVAE_s_no_q_hcf(
                config['input_shape'],
                num_embeddings=config['num_embeddings'],
                hcf_n = 30,
                codebook_size=config['codebook_size'],
                num_classes=config['num_classes'],
                diag_cov=config['diag_cov'],
                dropout = config['dropout']
            )
        else:
            model = VQVAE_s_no_q(
                config['input_shape'],
                num_embeddings=config['num_embeddings'],
                codebook_size=config['codebook_size'],
                num_classes=config['num_classes']
            )
    model = model.to(device)
    return model


def load_pretrained_model(model, config, device):
    """
    Load pretrained weights according to the configuration.
    Each component (encoder, classifier, and codebook) is loaded only if its
    corresponding flag is True. Then, if a freeze flag is set, the loaded part's 
    parameters are frozen. The file paths are chosen based on both the model_mode 
    and whether HCF is used.

    This version also handles 'TripletNet_hcf' style checkpoints that store
    CNN + HCF in 'embedding_net.*' and re-maps them to model.encoder, model.extra_fc, 
    and model.concat_fc as needed.
    """

    mode = config['model_mode']
    use_hcf = config['use_hcf']

    # -------------------------------------------------
    # Determine which file paths to load
    # -------------------------------------------------
    encoder_path = classifier_path = codebook_path = None

    if mode == 'close_set_enc':
        if use_hcf:
            encoder_path = config.get('encoder_weights_path_hcf_close_set_enc')
            classifier_path = config.get('classifier_weights_path_hcf_close_set_enc')
            codebook_path = config.get('codebook_weights_path_hcf_close_set_enc')
        else:
            encoder_path = config.get('encoder_weights_path_close_set_enc')
            classifier_path = config.get('classifier_weights_path_close_set_enc')
            codebook_path = config.get('codebook_weights_path_close_set_enc')

    elif mode == 'triplet':
        if use_hcf:
            encoder_path = config.get('encoder_weights_path_hcf_triplet')
            classifier_path = config.get('classifier_weights_path_hcf_triplet')
            codebook_path = config.get('codebook_weights_path_hcf_triplet')
        else:
            encoder_path = config.get('encoder_weights_path_triplet')
            classifier_path = config.get('classifier_weights_path_triplet')
            codebook_path = config.get('codebook_weights_path_triplet')

    elif mode == 'ss':
        if use_hcf:
            encoder_path = config.get('encoder_weights_path_hcf_ss')
            classifier_path = config.get('classifier_weights_path_hcf_ss')
            codebook_path = config.get('codebook_weights_path_hcf_ss')
        else:
            encoder_path = config.get('encoder_weights_path_ss')
            classifier_path = config.get('classifier_weights_path_ss')
            codebook_path = config.get('codebook_weights_path_ss')

    # -------------------------------------------------
    # 1) Load the encoder weights + HCF submodules
    # -------------------------------------------------
    if config['pretrained_encoder'] and encoder_path is not None and os.path.exists(encoder_path):
        state_dict = torch.load(encoder_path, map_location=device, weights_only=True)

        # We'll split the weights that belong to:
        #  - encoder (CNN)
        #  - extra_fc (HCF)
        #  - concat_fc (HCF)
        encoder_dict = {}
        extra_fc_dict = {}
        concat_fc_dict = {}
        old_classifier_dict = {}
        # Some old models store everything under "embedding_net.*"
        # Others might store them under "encoder.*"
        # We'll allow either prefix.
        possible_prefixes = ["embedding_net.", "encoder."]

        for key, val in state_dict.items():
            if key.startswith("quantizer.") or key.startswith("codebook."):
                continue
            # Strip off any recognized prefix
            sub_key = key
            for pfx in possible_prefixes:
                if sub_key.startswith(pfx):
                    sub_key = sub_key[len(pfx):]
                    break

            # Now figure out which part of the new model this belongs to
            if sub_key.startswith("extra_fc."):
                new_key = sub_key[len("extra_fc."):]  # e.g. "0.weight"
                extra_fc_dict[new_key] = val
            elif sub_key.startswith("concat_fc."):
                new_key = sub_key[len("concat_fc."):]
                concat_fc_dict[new_key] = val
            elif sub_key.startswith("classifier."):
                # <--- new logic to catch the old classifier
                new_key = sub_key[len("classifier."):]
                old_classifier_dict[new_key] = val
            else:
                # Everything else goes to the CNN encoder
                encoder_dict[sub_key] = val

        # Now load them into the new model
        # - The top-level new model has model.encoder, model.extra_fc, model.concat_fc
        #   if HCF is in use
        encoder_dict = {k: v for k, v in encoder_dict.items() if not k.startswith("norm.")}
        model.encoder.load_state_dict(encoder_dict, strict=True)
        print("Pretrained encoder loaded from", encoder_path)

        if use_hcf:
            model.extra_fc.load_state_dict(extra_fc_dict, strict=True)
            model.concat_fc.load_state_dict(concat_fc_dict, strict=True)
            print("Loaded HCF layers (extra_fc, concat_fc) from", encoder_path)
    else:
        if config['pretrained_encoder']:
            print("Encoder pretrained flag set but file not found:", encoder_path)

    # -------------------------------------------------
    # 2) Load the classifier weights
    # -------------------------------------------------
    if config['pretrained_classifier'] and classifier_path is not None and os.path.exists(classifier_path):
        classifier_state = torch.load(classifier_path, map_location=device, weights_only=True)
        # Typically, your new classifier is model.classifier, so let's parse that out
        # If the old checkpoint keys are "classifier.*", strip that prefix.
        classifier_dict = {
            k.replace("classifier.", ""): v
            for k, v in classifier_state.items()
            if k.startswith("classifier.")
        }
        model.classifier.load_state_dict(classifier_dict, strict=False)
        print("Pretrained classifier loaded from", classifier_path)
    else:
        if config['pretrained_classifier']:
            print("Classifier pretrained flag set but file not found:", classifier_path)

    # -------------------------------------------------
    # 3) Load codebook (if using a quantizer)
    # -------------------------------------------------
    # if config['use_quantizer'] and config['pretrained_codebook'] and hasattr(model, 'quantizer'):
    #     if codebook_path is not None and os.path.exists(codebook_path):
    #         #lbg_codebook = torch(codebook_path, map_location=device)
    #         lbg_codebook = torch.load(codebook_path, map_location=device)
    #         if isinstance(lbg_codebook, dict):
    #             if 'weight' in lbg_codebook:
    #                 model.quantizer.codebook.weight.data.copy_(lbg_codebook['weight'])
    #             else:
    #                 # Fallback: print keys for debugging
    #                 print(f"Loaded dict from codebook file, keys: {list(lbg_codebook.keys())}")
    #                 raise RuntimeError("Cannot find 'weight' key in codebook file.")
    #         else:
    #             model.quantizer.codebook.weight.data.copy_(lbg_codebook)

    #         # The codebook is just an nn.Embedding's weight
    #         model.quantizer.codebook.weight.data.copy_(lbg_codebook)
    #         print("Pretrained codebook loaded from", codebook_path)
    #     else:
    #         print("Codebook pretrained flag set but file not found:", codebook_path)
        # 3) Load codebook (if using a quantizer)
    if config['use_quantizer'] and config['pretrained_codebook'] and hasattr(model, 'quantizer'):
        if config.get('load_codebook_from_model_weights', False):
            # Try to load codebook from the encoder or classifier checkpoint instead of a separate file
            ckpt_path = encoder_path if encoder_path is not None and os.path.exists(encoder_path) else classifier_path
            if ckpt_path is not None and os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location=device)
                # Try both possible key names
                if 'quantizer.codebook.weight' in state_dict:
                    model.quantizer.codebook.weight.data.copy_(state_dict['quantizer.codebook.weight'])
                    print(f"Loaded codebook from model weights ({ckpt_path}, key: 'quantizer.codebook.weight')")
                elif 'codebook.weight' in state_dict:
                    model.quantizer.codebook.weight.data.copy_(state_dict['codebook.weight'])
                    print(f"Loaded codebook from model weights ({ckpt_path}, key: 'codebook.weight')")
                else:
                    print(f"Could not find codebook weights in {ckpt_path}. Available keys: {list(state_dict.keys())}")
            else:
                print("Codebook flag set to load from model weights, but checkpoint not found:", ckpt_path)
        else:
            # Default: load from codebook file
            if codebook_path is not None and os.path.exists(codebook_path):
                lbg_codebook = torch.load(codebook_path, map_location=device)
                if isinstance(lbg_codebook, dict):
                    if 'weight' in lbg_codebook:
                        model.quantizer.codebook.weight.data.copy_(lbg_codebook['weight'])
                    elif 'codebook.weight' in lbg_codebook:
                        model.quantizer.codebook.weight.data.copy_(lbg_codebook['codebook.weight'])
                    else:
                        print(f"Loaded dict from codebook file, keys: {list(lbg_codebook.keys())}")
                        raise RuntimeError("Cannot find codebook weight in codebook file.")
                else:
                    model.quantizer.codebook.weight.data.copy_(lbg_codebook)
                print("Pretrained codebook loaded from", codebook_path)
            else:
                print("Codebook pretrained flag set but file not found:", codebook_path)

    # -------------------------------------------------
    # 4) Freeze parameters if requested
    # -------------------------------------------------
    if config.get('freeze_encoder', False):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen.")
    if config.get('use_hcf', False) and config.get('freeze_hcf_params', False):
        for param in model.extra_fc.parameters():
            param.requires_grad = False
        for param in model.concat_fc.parameters():
            param.requires_grad = False
        print("HCF parameters frozen.")

    if config.get('freeze_classifier', False):
        for param in model.classifier.parameters():
            param.requires_grad = False
        print("Classifier parameters frozen.")

    if config['use_quantizer'] and config.get('freeze_codebook', False) and hasattr(model, 'quantizer'):
        for param in model.quantizer.parameters():
            param.requires_grad = False
        print("Codebook parameters frozen.")

    return model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For (slightly) more reproducible behavior, but might degrade performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    # Initialize configuration.
    config = setup_config()
    set_seed(config['seed'])
    # (Optional) Print config to verify settings.
    print("Training configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Set up Weights & Biases.
    wandb.login(key="f873a533ed8359c89bb63389601e9439b0f2f853")
    wandb.init(project="vq-vae-classifier", entity='https-www-bgu-ac-il-')
    
    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data loaders.
    train_loader, val_loader = setup_data(config)
    
    # Build the model.
    model = build_model(config, device)
    print("Instantiated model:", type(model))
    # Load pretrained parts as requested (and freeze parts if specified).
    model = load_pretrained_model(model, config, device)
    
    # Set optimizer and loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    # Choose the appropriate training loop based on whether HCF is used.
    if config['use_hcf']:
        train_losses, train_accuracies, val_losses, val_accuracies = train_fr_hcf(
            model, optimizer, criterion, train_loader, val_loader, device,
            epochs=config['epochs'], save_path=config['save_path']
        )
    else:
        train_losses, train_accuracies, val_losses, val_accuracies = train_fr(
            model, optimizer, criterion, train_loader, val_loader, device,
            epochs=config['epochs'], save_path=config['save_path']
        )
    
    print("Training completed.")

if __name__ == "__main__":
    main()