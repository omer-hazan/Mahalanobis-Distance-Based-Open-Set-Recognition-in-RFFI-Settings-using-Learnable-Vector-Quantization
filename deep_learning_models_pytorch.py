import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



#def estimate_class_mean_cov(embeddings, labels, num_classes):
#    means = []
#    covs = []
#    for cls in range(num_classes):
#        mask = (labels == cls)
#        emb_cls = embeddings[mask]
#        if emb_cls.size(0) < 2:
#            mean = emb_cls.mean(dim=0) if emb_cls.size(0) > 0 else torch.zeros(embeddings.size(1), device=embeddings.device)
#            cov = torch.eye(embeddings.size(1), device=embeddings.device)
#        else:
#            mean = emb_cls.mean(dim=0)
#            cov = torch.from_numpy(np.cov(emb_cls.cpu().detach().numpy(), rowvar=False)).float().to(embeddings.device)
#            cov += 1e-6 * torch.eye(cov.size(0), device=embeddings.device)
#        means.append(mean)
#        covs.append(cov)
#    return means, covs
 

def estimate_class_mean_cov(embeddings, labels, num_classes, diag_cov=False):
    # Flatten labels if needed
    labels = labels.view(-1)
    # Per-class means
    means = []
    for cls in range(num_classes):
        mask = (labels == cls)        # mask shape [N]
        emb_cls = embeddings[mask]    # [N_class, D]
        if emb_cls.size(0) == 0:
            mean = torch.zeros(embeddings.size(1), device=embeddings.device)
        else:
            mean = emb_cls.mean(dim=0)
        means.append(mean)
    # Global covariance
    if diag_cov:
        var = torch.from_numpy(
            np.var(embeddings.cpu().detach().numpy(), axis=0)
        ).float().to(embeddings.device)
        eps = 1e-6
        cov = torch.diag(var + eps)
    else:    
        cov = torch.from_numpy(
            np.cov(embeddings.cpu().detach().numpy(), rowvar=False)
        ).float().to(embeddings.device)
        cov += 1e-6 * torch.eye(cov.size(0), device=embeddings.device)
    return means, cov



def mahalanobis_distance(x, y, cov_inv):
    """
    x, y: shape (B, D)
    cov_inv: (D, D) or (D,)
    Returns: shape (B,)
    """
    diff = x - y
    if cov_inv.dim() == 2:
        left = torch.matmul(diff, cov_inv)
        dist_sq = (left * diff).sum(dim=1)
    else:
        dist_sq = (diff * cov_inv * diff).sum(dim=1)
    return torch.sqrt(F.relu(dist_sq) + 1e-12)

class MahalanobisTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, cov_inv):
        # All inputs: (B, D)
        d_ap = mahalanobis_distance(anchor, positive, cov_inv)
        d_an = mahalanobis_distance(anchor, negative, cov_inv)
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()



def class_mean(embeddings, labels, class_idx):
    mask = (labels == class_idx)
    class_embs = embeddings[mask]
    return class_embs.mean(dim=0) if class_embs.size(0) > 0 else None

class CentroidTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, labels, embeddings, cov_inv):
        batch_size = anchor.size(0)
        losses = []
        for i in range(batch_size):
            anchor_class = labels[i].item()
            anchor_embedding = anchor[i]

            # Positive: mean of all other anchor_class in batch (excluding anchor)
            positive_mean = class_mean_except_anchor(embeddings, labels, anchor_class, i)
            pos_dist = mahalanobis_distance(anchor_embedding, positive_mean, cov_inv)

            # Negative: mean of a different class, e.g., random other class in batch
            negative_classes = set(labels.cpu().numpy()) - {anchor_class}
            # For example, pick one negative class at random:
            negative_class = random.choice(list(negative_classes))
            negative_mean = class_mean(embeddings, labels, negative_class)
            neg_dist = mahalanobis_distance(anchor_embedding, negative_mean, cov_inv)

            loss = F.relu(pos_dist - neg_dist + self.margin)
            losses.append(loss)
        return torch.stack(losses).mean()



# Residual block definition
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

# Identity loss
def identity_loss(y_true, y_pred):
    return torch.mean(y_pred)

# TripletNet definition
class TripletNet(nn.Module):
    def __init__(self, datashape, alpha):
        super(TripletNet, self).__init__()
        self.datashape = datashape
        self.alpha = alpha
        self.embedding_net = self.feature_extractor()

    def triplet_loss(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        basic_loss = pos_dist - neg_dist + self.alpha
        # loss = torch.clamp(basic_loss, min=0.0)
        loss = basic_loss
        return loss.mean()

    def feature_extractor(self):
        class FeatureExtractor(nn.Module):
            def __init__(self, datashape):
                super(FeatureExtractor, self).__init__()
                self.conv1 = nn.Conv2d(datashape[1], 32, 7, stride=2, padding=3)
                self.resblock1 = ResBlock(32, 32, 3)
                self.resblock2 = ResBlock(32, 32, 3)
                self.resblock3 = ResBlock(32, 64, 3, first_layer=True)
                self.resblock4 = ResBlock(64, 64, 3)
                self.pool = nn.AvgPool2d(2)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(24000, 128)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.resblock1(x)
                x = self.resblock2(x)
                x = self.resblock3(x)
                x = self.resblock4(x)
                x = self.pool(x)
                x = self.flatten(x)
                x = self.fc(x)
                x = F.normalize(x, p=2, dim=1)
                return x

        return FeatureExtractor(self.datashape)

    def forward(self, input_1, input_2, input_3):
        anchor = self.embedding_net(input_1)
        positive = self.embedding_net(input_2)
        negative = self.embedding_net(input_3)
        loss = self.triplet_loss(anchor, positive, negative)
        return loss

    def create_generator(self, batchsize, dev_range, data, label):
        self.data = data
        self.label = label
        self.dev_range = dev_range

#        def get_triplet():
#            n = a = np.random.choice(self.dev_range)
#            while n == a:
#                n = np.random.choice(self.dev_range)
#            a, p = call_sample(a), call_sample(a)
#            n = call_sample(n)
#            return a, p, n

        def call_sample(label_name):
            num_sample = len(self.label)
            idx = np.random.randint(num_sample)
            while self.label[idx] != label_name:
                idx = np.random.randint(num_sample)
            return self.data[idx]
        
        def get_triplet():
            n = a = np.random.choice(self.dev_range)
            while n == a:
                n = np.random.choice(self.dev_range)
            # Find indices for label==a and label==n
            idx_a = np.random.choice(np.where(label == a)[0])
            idx_p = np.random.choice(np.where(label == a)[0])
            idx_n = np.random.choice(np.where(label == n)[0])
            return data[idx_a], data[idx_p], data[idx_n], a  # <- return anchor label
    
        while True:
            list_a, list_p, list_n, list_labels = [], [], [], []
            for _ in range(batchsize):
                a, p, n, lbl = get_triplet()
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)
                list_labels.append(lbl)
            A = torch.tensor(np.array(list_a, dtype='float32'))
            P = torch.tensor(np.array(list_p, dtype='float32'))
            N = torch.tensor(np.array(list_n, dtype='float32'))
            L = torch.tensor(np.array(list_labels, dtype='int64'))
            yield [A, P, N, L], torch.ones(batchsize)
        
        
#        while True:
#            list_a, list_p, list_n = [], [], []
#            for _ in range(batchsize):
#                a, p, n = get_triplet()
#                list_a.append(a)
#                list_p.append(p)
#                list_n.append(n)
#
#            A = torch.tensor(np.array(list_a, dtype='float32'))
#            P = torch.tensor(np.array(list_p, dtype='float32'))
#            N = torch.tensor(np.array(list_n, dtype='float32'))
#            yield [A, P, N], torch.ones(batchsize)
      
      

      
        
        
class TripletNet_hcf(nn.Module):
    def __init__(self, datashape, alpha):
        super().__init__()
        self.alpha = alpha
        self.embedding_net = self._build_feature_extractor(datashape)

    # ---------------- triplet loss (unchanged) ----------------
    def triplet_loss(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        return torch.clamp(pos_dist - neg_dist + self.alpha, min=0).mean()

    # ---------------- feature extractor ----------------
    def _build_feature_extractor(self, datashape):
        class FeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                C = datashape[1]
                self.conv1 = nn.Conv2d(C, 32, kernel_size=7, stride=2, padding=3)
                self.resblock1 = ResBlock(32, 32, 3)
                self.resblock2 = ResBlock(32, 32, 3)
                self.resblock3 = ResBlock(32, 64, 3, first_layer=True)
                self.resblock4 = ResBlock(64, 64, 3)
                self.pool      = nn.AvgPool2d(2)
                self.fc   = nn.Linear(24000, 128)

                # HCF branch
                self.extra_fc  = nn.Sequential(nn.Linear(3, 30), nn.ReLU())
                self.concat_fc = nn.Sequential(nn.Linear(128 + 30, 128), nn.ReLU())

            def forward(self, x, extra, *, return_latents=False):
                x = F.relu(self.conv1(x))
                x = self.resblock1(x)
                x = self.resblock2(x)
                x = self.resblock3(x)
                x = self.resblock4(x)
                x = self.pool(x)
                x = self.fc(torch.flatten(x, 1))

                extra = self.extra_fc(extra)
                combined = self.concat_fc(torch.cat([x, extra], dim=1))
                combined = F.normalize(combined, p=2, dim=1)  # final embedding

                if return_latents:
                    return combined
                else:
                    return combined  # kept for compatibility

        return FeatureExtractor()

    # ---------------- public helpers ----------------
    def encode(self, spectrogram, extra):
        """Return *only* the L2‑normalised embedding."""
        return self.embedding_net(spectrogram, extra, return_latents=True)

    # standard forward keeps the triplet‑loss behaviour
    def forward(self, anchor, positive, negative, extra_a, extra_p, extra_n):
        emb_a = self.embedding_net(anchor,   extra_a, return_latents=True)
        emb_p = self.embedding_net(positive, extra_p, return_latents=True)
        emb_n = self.embedding_net(negative, extra_n, return_latents=True)
        return self.triplet_loss(emb_a, emb_p, emb_n)

    def forward(self, input_1, extra_1, input_2, extra_2, input_3, extra_3):
        """
        Args:
            input_1, input_2, input_3 (Tensor): CNN inputs (e.g. spectrogram images) for anchor, positive, and negative.
            extra_1, extra_2, extra_3 (Tensor): Hand-crafted feature vectors (shape: [batch, 3]) for anchor, positive, and negative.
        Returns:
            loss (Tensor): Triplet loss computed over the batch.
        """
        anchor = self.embedding_net(input_1, extra_1)
        positive = self.embedding_net(input_2, extra_2)
        negative = self.embedding_net(input_3, extra_3)
        loss = self.triplet_loss(anchor, positive, negative)
        return loss

    def create_generator(self, batchsize, dev_range, data, extra_features, label):
        """
        A generator yielding triplets for training.
        Args:
            batchsize (int): Number of triplets per batch.
            dev_range (iterable): Device range (or label range) to sample from.
            data (np.array): CNN input data (e.g. spectrograms).
            extra_features (np.array): Extra hand-crafted features corresponding to data.
            label (np.array): Labels corresponding to data.
        Yields:
            A tuple: ([A, extra_A, P, extra_P, N, extra_N], target) where each element is a Tensor.
        """
        self.data = data
        self.extra_features = extra_features
        self.label = label
        self.dev_range = dev_range

        def get_triplet():
            a = np.random.choice(self.dev_range)
            n = a
            while n == a:
                n = np.random.choice(self.dev_range)
            a_sample = call_sample(a)
            p_sample = call_sample(a)
            n_sample = call_sample(n)
            return a_sample, p_sample, n_sample

        def call_sample(class_id):
            idxs = np.where(self.label == class_id)[0]
            idx = np.random.choice(idxs)
            # Should return three elements:
            # self.data[idx], self.extra_features[idx], self.label[idx]
            return self.data[idx], self.extra_features[idx], self.label[idx]

        while True:
            list_a_img, list_p_img, list_n_img = [], [], []
            list_a_extra, list_p_extra, list_n_extra = [], [], []
            list_a_label, list_p_label, list_n_label = [], [], []
            for _ in range(batchsize):
                a_class = np.random.choice(self.dev_range)
                n_class = a_class
                while n_class == a_class:
                    n_class = np.random.choice(self.dev_range)
                a_img, a_extra, a_lbl = call_sample(a_class)
                p_img, p_extra, p_lbl = call_sample(a_class)
                n_img, n_extra, n_lbl = call_sample(n_class)
                list_a_img.append(a_img)
                list_p_img.append(p_img)
                list_n_img.append(n_img)
                list_a_extra.append(a_extra)
                list_p_extra.append(p_extra)
                list_n_extra.append(n_extra)
                list_a_label.append(a_lbl)
                list_p_label.append(p_lbl)
                list_n_label.append(n_lbl)
        
            # Stack
            A = torch.tensor(np.array(list_a_img, dtype='float32'))
            extra_A = torch.tensor(np.array(list_a_extra, dtype='float32'))
            P = torch.tensor(np.array(list_p_img, dtype='float32'))
            extra_P = torch.tensor(np.array(list_p_extra, dtype='float32'))
            N = torch.tensor(np.array(list_n_img, dtype='float32'))
            extra_N = torch.tensor(np.array(list_n_extra, dtype='float32'))
            anchor_labels = torch.tensor(np.array(list_a_label, dtype='int64'))
            pos_labels = torch.tensor(np.array(list_p_label, dtype='int64'))
            neg_labels = torch.tensor(np.array(list_n_label, dtype='int64'))
            yield [A, extra_A, P, extra_P, N, extra_N], (anchor_labels, pos_labels, neg_labels)
        
