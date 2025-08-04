import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import time

import os
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import math


from liverpool.closeset.dataset_preparation import LoadDataset, awgn, ChannelIndSpectrogram

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################################
#                          Residual + Feature Extractors                       #
################################################################################

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, first_layer=False):
        super(ResBlock, self).__init__()

        self.first_layer = first_layer
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if self.first_layer or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FeatureExtractor(nn.Module):
    """
    Smaller feature extractor (used in VQVAE_s or VQVAE_s_hcf).
    """
    def __init__(self, input_shape, num_embeddings):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2, padding=3)
        self.resblock1 = ResBlock(32, 32)
        self.resblock2 = ResBlock(32, 32)
        self.resblock3 = ResBlock(32, 64, first_layer=True)
        self.resblock4 = ResBlock(64, 64)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # The 24000 in fc might need adjusting if your input dims change
        self.fc = nn.Linear(24000, num_embeddings)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        return x


class FeatureExtractor18(nn.Module):
    """
    Larger feature extractor (used in VQVAE, VQVAE_no_q).
    """
    def __init__(self, input_shape):
        super(FeatureExtractor18, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2, padding=3)
        self.resblock1 = ResBlock(32, 32)
        self.resblock2 = ResBlock(32, 64)
        self.resblock3 = ResBlock(64, 64, first_layer=True)
        self.resblock4 = ResBlock(64, 128)
        self.resblock5 = ResBlock(128, 128, first_layer=True)
        self.resblock6 = ResBlock(128, 256)
        self.resblock7 = ResBlock(256, 256, first_layer=True)
        self.resblock8 = ResBlock(256, 256)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # The 96000 in fc might need adjusting if your input dims change
        self.fc = nn.Linear(96000, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        return x


################################################################################
#                          Fixed-Rate Vector Quantizer                         #
################################################################################

#class FixedRateVectorQuantizer(nn.Module):
#    def __init__(self, num_embeddings: int, codebook_size: int, commitment_loss_weight=0.1):
#        """
#        Args:
#            num_embeddings (int): Dimension of each embedding vector (z_e).
#            codebook_size (int): Number of vectors in the codebook (p).
#        """
#        super(FixedRateVectorQuantizer, self).__init__()
#        self.d = num_embeddings            # Dimension of each latent vector
#        self.p = codebook_size            # Number of codebook entries
#        self.commitment_loss_weight = commitment_loss_weight
#
#        # Initialize the codebook
#        self.codebook = nn.Embedding(self.p, self.d)
#        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)
#
#    def forward(self, input_data):
#        """
#        Returns: (quantized, commitment_loss, codebook_loss, encoding_indices)
#        """
#        # Calculate distances between input vectors and codebook vectors
#        # input_data: shape [B, d]
#        # codebook.weight: shape [p, d]
#
#        # (B, 1) + (p,) - 2*(B, d)*(d, p) => (B, p)
#        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
#                     + torch.sum(self.codebook.weight ** 2, dim=1)
#                     - 2 * torch.matmul(input_data, self.codebook.weight.t()))
#
#        # Encoding: find index of closest codebook vector for each input
#        encoding_indices = torch.argmin(distances, dim=1)  # shape: [B]
#        encodings = F.one_hot(encoding_indices, num_classes=self.p).float()  # [B, p]
#
#        # Quantize
#        quantized = torch.matmul(encodings, self.codebook.weight)  # [B, d]
#
#        # Codebook loss and commitment loss
#        codebook_loss = F.mse_loss(quantized.detach(), input_data)
#        commitment_loss = self.commitment_loss_weight * F.mse_loss(quantized, input_data.detach())
#
#        # Straight-through estimator
#        quantized = input_data + (quantized - input_data).detach()
#
#        return quantized, commitment_loss, codebook_loss, encoding_indices





class FixedRateVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, codebook_size: int, commitment_loss_weight=0.1,diag_cov=False,entropy_lambda=200000):
        """
        Args:
            num_embeddings (int): Dimension of each embedding vector (z_e).
            codebook_size (int): Number of vectors in the codebook (p).
        """
        super(FixedRateVectorQuantizer, self).__init__()
        self.d = num_embeddings            # Dimension of each latent vector
        self.p = codebook_size             # Number of codebook entries
        self.commitment_loss_weight = commitment_loss_weight
        self.entropy_lambda=entropy_lambda
        # Initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)
        self.diag_cov = diag_cov
    def get_cov_inv(self):
        # Compute the inverse covariance matrix of the codebook weights
        # Add regularization for numerical stability
        with torch.no_grad():
            mu = self.codebook.weight.mean(dim=0, keepdim=True)  # [1, d]
            centered = self.codebook.weight - mu                 # [p, d]
            cov = centered.t() @ centered / (self.p - 1)         # [d, d]
            cov += 1e-3 * torch.eye(self.d, device=cov.device)   # Regularization
            inv_cov = torch.inverse(cov)                         # [d, d]
        return inv_cov

    def get_diag_cov_inv(self):
    # Compute the inverse of the diagonal covariance matrix of the codebook weights
    # Add regularization for numerical stability
        with torch.no_grad():
            mu = self.codebook.weight.mean(dim=0, keepdim=True)  # [1, d]
            centered = self.codebook.weight - mu                 # [p, d]
            var = centered.pow(2).mean(dim=0)                   # [d]
            # var += 1e-3                                         # Regularization
            inv_diag_cov = 1.0 / var                            # [d]
            inv_cov = torch.diag(inv_diag_cov)                  # [d, d] (diagonal matrix)
        return inv_cov

    def mahalanobis_distances(self, input_data, inv_cov):
        # input_data: [B, d]
        # codebook.weight: [p, d]
        # inv_cov: [d, d]
        # Output: [B, p] - Mahalanobis distance between each input and each codebook vector

        x = input_data.unsqueeze(1)      # [B, 1, d]
        mu = self.codebook.weight.unsqueeze(0)  # [1, p, d]
        diff = x - mu                    # [B, p, d]
        left = torch.matmul(diff, inv_cov)    # [B, p, d]
        mahal_sq = (left * diff).sum(dim=2)   # [B, p]
        return mahal_sq
    
    def batch_mahalanobis(self, x, y, inv_cov):
        diff = x - y
        left = torch.matmul(diff, inv_cov)
        mahal_sq = (left * diff).sum(dim=1)
        return mahal_sq.mean()

    def forward(self, input_data):
        """
        Returns: (quantized, commitment_loss, codebook_loss, encoding_indices)
        """
        if self.diag_cov:
            inv_cov = self.get_diag_cov_inv()
        else:
            inv_cov = self.get_cov_inv()  # [d, d]
        
        distances = self.mahalanobis_distances(input_data, inv_cov)  # [B, p]
        
        # Encoding: find index of closest codebook vector for each input
        encoding_indices = torch.argmin(distances, dim=1)  # shape: [B]
        encodings = F.one_hot(encoding_indices, num_classes=self.p).float()  # [B, p]
        
        # Quantize
        quantized = torch.matmul(encodings, self.codebook.weight)  # [B, d]
        
        # Codebook loss and commitment loss
        # Mahalanobis losses:
        codebook_loss = self.batch_mahalanobis(quantized.detach(), input_data, inv_cov)
        commitment_loss = self.commitment_loss_weight * self.batch_mahalanobis(quantized, input_data.detach(), inv_cov)
        
        # ===== ENTROPY REGULARIZATION =====
        usage_counts = encodings.sum(dim=0)  # [p]
        usage_probs = usage_counts / (usage_counts.sum() + 1e-8)  # [p]
        entropy = -torch.sum(usage_probs * torch.log(usage_probs + 1e-8))
        entropy_reg = -self.entropy_lambda * entropy  # maximize entropy by minimizing -entropy
        #print("entropy_reg",entropy_reg)
        #print("entropy lambda",self.entropy_lambda)
        codebook_loss = codebook_loss + entropy_reg
        
        # Straight-through estimator
        quantized = input_data + (quantized - input_data).detach()
        
        return quantized, commitment_loss, codebook_loss, encoding_indices







################################################################################
#                  VQVAE / VQVAE_s + (optional) HCF + no-q variants            #
################################################################################

class VQVAE(nn.Module):
    """
    Uses the larger FeatureExtractor18 + quantizer + classifier.
    """
    def __init__(self, input_shape, num_embeddings, codebook_size, num_classes, commitment_loss_weight=0.1):
        super(VQVAE, self).__init__()
        self.encoder = FeatureExtractor18(input_shape)
        self.quantizer = FixedRateVectorQuantizer(num_embeddings=512, 
                                                  codebook_size=codebook_size,
                                                  commitment_loss_weight=commitment_loss_weight)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512//2),
            nn.ReLU(),
            nn.Linear(512//2, 512//4),
            nn.ReLU(),
            nn.Linear(512//4, num_classes)
        )

    def get_accuracy(self, gt, preds):
        pred_vals = torch.argmax(preds, dim=1)
        batch_correct = (pred_vals == gt).sum().item()
        return batch_correct

    def forward(self, inputs):
        # Encode
        z_e = self.encoder(inputs)  # shape [B, 512]

        # Vector-Quantize
        z_q, commit_loss, codebook_loss, code_indices = self.quantizer(z_e)

        # Classify
        class_logits = self.classifier(z_q)

        # Combine losses
        quantization_loss = commit_loss + codebook_loss

        return class_logits, quantization_loss, code_indices


class VQVAE_s(nn.Module):
    """
    Uses the smaller FeatureExtractor + quantizer + classifier.
    """
    def __init__(self, input_shape, num_embeddings, codebook_size, num_classes, commitment_loss_weight=0.1):
        super(VQVAE_s, self).__init__()

        self.encoder = FeatureExtractor(input_shape, num_embeddings)
        self.quantizer = FixedRateVectorQuantizer(num_embeddings=num_embeddings,
                                                  codebook_size=codebook_size,
                                                  commitment_loss_weight=commitment_loss_weight)
        self.classifier = nn.Sequential(
            nn.Linear(num_embeddings, num_embeddings//2),
            nn.ReLU(),
            nn.Linear(num_embeddings//2, num_embeddings//4),
            nn.ReLU(),
            nn.Linear(num_embeddings//4, num_classes)
        )

    def get_accuracy(self, gt, preds):
        pred_vals = torch.argmax(preds, dim=1)
        batch_correct = (pred_vals == gt).sum().item()
        return batch_correct

    def forward(self, inputs):
        z_e = self.encoder(inputs)
        z_q, commit_loss, codebook_loss, code_indices = self.quantizer(z_e)
        class_logits = self.classifier(z_q)

        quantization_loss = commit_loss + codebook_loss
        return class_logits, quantization_loss, code_indices


class VQVAE_no_q(nn.Module):
    """
    Uses the larger FeatureExtractor18, but NO quantizer. Just pass z_e directly.
    """
    def __init__(self, input_shape, num_embeddings, codebook_size, num_classes, commitment_loss_weight=0.1):
        super(VQVAE_no_q, self).__init__()

        self.encoder = FeatureExtractor18(input_shape)
        # no quantizer
        self.classifier = nn.Sequential(
            nn.Linear(512, 512//2),
            nn.ReLU(),
            nn.Linear(512//2, 512//4),
            nn.ReLU(),
            nn.Linear(512//4, num_classes)
        )

    def get_accuracy(self, gt, preds):
        pred_vals = torch.argmax(preds, dim=1)
        batch_correct = (pred_vals == gt).sum().item()
        return batch_correct

    def forward(self, inputs):
        z_e = self.encoder(inputs)
        class_logits = self.classifier(z_e)
        # quant_loss is None
        return class_logits, None, None


class VQVAE_s_no_q(nn.Module):
    """
    Uses the smaller FeatureExtractor, but NO quantizer.
    """
    def __init__(self, input_shape, num_embeddings, codebook_size, num_classes, commitment_loss_weight=0.1):
        super(VQVAE_s_no_q, self).__init__()

        self.encoder = FeatureExtractor(input_shape, num_embeddings)
        # no quantizer
        self.classifier = nn.Sequential(
            nn.Linear(num_embeddings, num_embeddings//2),
            nn.ReLU(),
            nn.Linear(num_embeddings//2, num_embeddings//4),
            nn.ReLU(),
            nn.Linear(num_embeddings//4, num_classes)
        )

    def get_accuracy(self, gt, preds):
        pred_vals = torch.argmax(preds, dim=1)
        batch_correct = (pred_vals == gt).sum().item()
        return batch_correct

    def forward(self, inputs):
        z_e = self.encoder(inputs)
        class_logits = self.classifier(z_e)
        return class_logits, None, None


################################################################################
#                  HCF (hand-crafted features) variants                        #
################################################################################

class VQVAE_s_no_q_hcf(nn.Module):
    """
    VQ‑VAE_s without a quantiser, but with the HCF branch.
    When `return_latents=True` the forward pass also returns the
    latent representation used by the classifier.
    """
    def __init__(self,
                 input_shape,
                 num_embeddings,
                 hcf_n,
                 codebook_size,          # kept for API symmetry; not used here
                 num_classes,
                 commitment_loss_weight=0.1,
                 diag_cov=False,
                 dropout=False,
                 dropout_p=0.3):
        super().__init__()

        # Main encoder that produces z_e
        self.encoder = FeatureExtractor(input_shape, num_embeddings)
        self.norm = nn.LayerNorm(num_embeddings)

        # Three handcrafted‑feature inputs → hidden dim hcf_n
        self.extra_fc = nn.Sequential(
            nn.Linear(3, hcf_n),
            nn.ReLU()
        )

        # Concatenate z_e and HCF branch, then reduce back to num_embeddings
        self.concat_fc = nn.Sequential(
            nn.Linear(num_embeddings + hcf_n, num_embeddings),
            nn.ReLU()
        )
        if dropout:
             # Simple MLP classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_embeddings, num_embeddings // 2),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(num_embeddings // 2, num_embeddings // 4),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(num_embeddings // 4, num_classes)
            )
        else:
            # Simple MLP classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_embeddings, num_embeddings // 2),
                nn.ReLU(),
                nn.Linear(num_embeddings // 2, num_embeddings // 4),
                nn.ReLU(),
                nn.Linear(num_embeddings // 4, num_classes)
            )

    # convenience helper (unchanged)
    def get_accuracy(self, gt, preds):
        pred_vals = torch.argmax(preds, dim=1)
        return (pred_vals == gt).sum().item()

    # -------------  Forward pass  -------------
    def forward(self, inputs, extra_features, *, return_latents: bool = False):
        """
        Parameters
        ----------
        inputs : Tensor
            Main input tensor (e.g. spectrogram / IQ features) – shape [B, …].
        extra_features : Tensor
            Hand‑crafted feature tensor – expected shape [B, 3].
        return_latents : bool, optional (default False)
            If True, also return the latent vector fed to the classifier.

        Returns
        -------
        tuple
            • class_logits  : Tensor, shape [B, num_classes]  
            • quant_loss    : None                            (no quantiser)  
            • code_indices  : None                            (no quantiser)  
            • combined      : Tensor, shape [B, num_embeddings] (only if
              return_latents=True)
        """
        # Encoder branch
        z_e = self.encoder(inputs)                 # [B, num_embeddings]

        # Hand‑crafted‑feature branch
        extra_out = self.extra_fc(extra_features)  # [B, hcf_n]

        # Merge & fuse
        combined = torch.cat([z_e, extra_out], dim=1)  # [B, num_emb + hcf_n]
        combined = self.concat_fc(combined)            # [B, num_embeddings]
        combined = F.normalize(combined, p=2, dim=1)

        # Classification head
        class_logits = self.classifier(combined)

        if return_latents:
            return class_logits, None, None, combined
        else:
            return class_logits, None, None


# class VQVAE_s_hcf(nn.Module):
#     """
#     VQVAE_s with HCF. Includes a quantizer and extra feature branch.
#     """
#     def __init__(self, input_shape, num_embeddings, hcf_n, codebook_size, num_classes, commitment_loss_weight=0.01):
#         super(VQVAE_s_hcf, self).__init__()

#         self.encoder = FeatureExtractor(input_shape, num_embeddings)

#         self.extra_fc = nn.Sequential(
#             nn.Linear(3, hcf_n),
#             nn.ReLU()
#         )
#         self.concat_fc = nn.Sequential(
#             nn.Linear(num_embeddings + hcf_n, num_embeddings),
#             nn.ReLU()
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(num_embeddings, num_embeddings//2),
#             nn.ReLU(),
#             nn.Linear(num_embeddings//2, num_embeddings//4),
#             nn.ReLU(),
#             nn.Linear(num_embeddings//4, num_classes)
#         )

#         self.quantizer = FixedRateVectorQuantizer(num_embeddings=num_embeddings,
#                                                   codebook_size=codebook_size,
#                                                   commitment_loss_weight=commitment_loss_weight)

#     def get_accuracy(self, gt, preds):
#         pred_vals = torch.argmax(preds, dim=1)
#         batch_correct = (pred_vals == gt).sum().item()
#         return batch_correct

#     def forward(self, inputs, extra_features):
#         # Encode + HCF
#         z_e = self.encoder(inputs)              # shape: [B, num_embeddings]
#         extra_out = self.extra_fc(extra_features)  # shape: [B, hcf_n]
#         combined = torch.cat([z_e, extra_out], dim=1)  # shape: [B, num_embeddings + hcf_n]
#         combined = self.concat_fc(combined)     # shape: [B, num_embeddings]

#         # Quantize
#         z_q, commit_loss, codebook_loss, code_indices = self.quantizer(combined)

#         # Classify
#         class_logits = self.classifier(z_q)

#         # Combine losses
#         quantization_loss = commit_loss + codebook_loss

#         return class_logits, quantization_loss, code_indices

class VQVAE_s_hcf(nn.Module):
    """
    VQVAE_s with HCF. Includes a quantizer and extra feature branch.
    """
    def __init__(self, input_shape, num_embeddings, hcf_n, codebook_size, num_classes, commitment_loss_weight=0.1, diag_cov=False,dropout=False,dropout_p=0.3,entropy_lambda=0): #lambda from hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
        super(VQVAE_s_hcf, self).__init__()

        self.encoder = FeatureExtractor(input_shape, num_embeddings)
        self.norm = nn.LayerNorm(num_embeddings)
        self.entropy_lambda=entropy_lambda
        self.extra_fc = nn.Sequential(
            nn.Linear(3, hcf_n),
            nn.ReLU()
        )
        self.concat_fc = nn.Sequential(
            nn.Linear(num_embeddings + hcf_n, num_embeddings),
            nn.ReLU()
        )
        if dropout:
            self.classifier = nn.Sequential(
                nn.Linear(num_embeddings, num_embeddings//2),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(num_embeddings//2, num_embeddings//4),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(num_embeddings//4, num_classes)
            )
        else:
                self.classifier = nn.Sequential(
                nn.Linear(num_embeddings, num_embeddings//2),
                nn.ReLU(),
                nn.Linear(num_embeddings//2, num_embeddings//4),
                nn.ReLU(),
                nn.Linear(num_embeddings//4, num_classes)
            )
        
        self.quantizer = FixedRateVectorQuantizer(
            num_embeddings=num_embeddings,
            codebook_size=codebook_size,
            commitment_loss_weight=commitment_loss_weight,
            entropy_lambda=entropy_lambda
        )

    def get_accuracy(self, gt, preds):
        pred_vals = torch.argmax(preds, dim=1)
        batch_correct = (pred_vals == gt).sum().item()
        return batch_correct

    def forward(self, inputs, extra_features, return_latents=False):
        """
        If return_latents = True, we return an additional item (z_q) 
        for silhouette or other analyses.
        """
        # Encode + HCF
        z_e = self.encoder(inputs)                   # shape: [B, num_embeddings]
        extra_out = self.extra_fc(extra_features)    # shape: [B, hcf_n]
        combined = torch.cat([z_e, extra_out], dim=1)# shape: [B, num_embeddings + hcf_n]
        combined = self.concat_fc(combined)          # shape: [B, num_embeddings]
        combined = F.normalize(combined, p=2, dim=1)

        # Quantize
        z_q, commit_loss, codebook_loss, code_indices = self.quantizer(combined)

        # Classify
        class_logits = self.classifier(z_q)

        # Combine losses
        quantization_loss = commit_loss + codebook_loss

        if return_latents:
            return class_logits, quantization_loss, code_indices, combined
        else:
            return class_logits, quantization_loss, code_indices

