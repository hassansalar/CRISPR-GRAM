import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Enhanced Supervised Contrastive Loss
    """
    def __init__(self, temperature=0.08, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.size(0)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        similarity = torch.matmul(embeddings, embeddings.T)
        similarity = similarity / self.temperature
        similarity = similarity - similarity.max(dim=1, keepdim=True)[0]

        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)
        diagonal = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - diagonal

        exp_similarity = torch.exp(similarity) * (1 - diagonal)
        denominator = exp_similarity.sum(dim=1, keepdim=True) + self.eps

        log_prob = similarity - torch.log(denominator)

        positive_count = positive_mask.sum(dim=1)
        valid_mask = positive_count > 0
        
        if valid_mask.any():
            loss = -(positive_mask * log_prob).sum(dim=1) / (positive_count + self.eps)
            return loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=device)


class TripletMarginLoss(nn.Module):
    """
    Triplet loss with adaptive margin
    """
    def __init__(self, margin=0.5, eps=1e-8):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.size(0)
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        dist_matrix = 2 - 2 * torch.matmul(embeddings, embeddings.T)
        
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)
        negative_mask = 1 - positive_mask
        
        diagonal = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - diagonal
        negative_mask = negative_mask - diagonal
        
        pos_dist = (dist_matrix * positive_mask).max(dim=1)[0]
        neg_dist = (dist_matrix * negative_mask + 1000 * (1 - negative_mask)).min(dim=1)[0]
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        
        valid_mask = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)
        
        if valid_mask.any():
            return loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=device)


class ManifoldContrastiveLoss(nn.Module):
    """
    Combined contrastive and triplet loss
    """
    def __init__(self, temperature=0.08, triplet_margin=0.5):
        super().__init__()
        self.contrastive = SupervisedContrastiveLoss(temperature)
        self.triplet = TripletMarginLoss(triplet_margin)

    def forward(self, embeddings, labels):
        cont_loss = self.contrastive(embeddings, labels)
        trip_loss = self.triplet(embeddings, labels)
        return cont_loss + 0.5 * trip_loss


class CRISPRLoss(nn.Module):
    """
    Total Loss: L = BCE + lambda * L_manifold + variance_regularization
    """
    def __init__(self, manifold_weight=0.15, temperature=0.08, triplet_margin=0.5):
        super().__init__()

        self.cls_loss = nn.BCEWithLogitsLoss()
        self.manifold_loss = ManifoldContrastiveLoss(temperature, triplet_margin)
        self.manifold_weight = manifold_weight

    def forward(self, logits, projection, labels):
        labels_float = labels.float()
        labels_long = labels.long()

        classification = self.cls_loss(logits.squeeze(-1), labels_float)
        manifold = self.manifold_loss(projection, labels_long)

        class_0 = projection[labels == 0]
        class_1 = projection[labels == 1]
        
        variance_loss = 0
        if len(class_0) > 1:
            variance_loss += torch.var(class_0, dim=0).mean()
        if len(class_1) > 1:
            variance_loss += torch.var(class_1, dim=0).mean()
        
        total = classification + self.manifold_weight * manifold + 0.01 * variance_loss

        return {
            "total_loss": total,
            "classification_loss": classification,
            "manifold_loss": manifold,
            "variance_loss": variance_loss
        }
