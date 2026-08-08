import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_projection import InputEmbedding


class MultiHeadGraphAttention(nn.Module):
    """
    Residual Multi-head Graph Attention
    """
    def __init__(self, dim=128, num_heads=8, dropout=0.12):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)

        B, N, D = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)

        out = attention @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class GraphEncoder(nn.Module):
    """
    Stacked graph attention blocks
    """
    def __init__(self, dim=128, layers=4, heads=8, dropout=0.12):
        super().__init__()

        self.layers = nn.ModuleList([
            MultiHeadGraphAttention(dim, heads, dropout) 
            for _ in range(layers)
        ])
        
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class AttentionPooling(nn.Module):
    """
    Attention-based graph pooling
    """
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        
        self.multi_head_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        x, _ = self.multi_head_attn(x, x, x)
        
        attention = self.score(x)
        attention = torch.softmax(attention, dim=1)
        pooled = torch.sum(attention * x, dim=1)
        return pooled


class MeanPooling(nn.Module):
    """
    Simple mean pooling (برای ablation)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1)


class ManifoldEncoder(nn.Module):
    """
    Geometry-aware manifold encoder
    """
    def __init__(self, input_dim=128, manifold_dim=96, dropout=0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, manifold_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=1)
        return z


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    """
    def __init__(self, input_dim=96, projection_dim=64):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        return F.normalize(self.projector(x), p=2, dim=1)


class ClassificationHead(nn.Module):
    """
    Enhanced classification head with dynamic input dimension
    """
    def __init__(self, input_dim=96, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        logits = self.classifier(x)
        probability = torch.sigmoid(logits)
        return logits, probability


class CRISPR_GRAM(nn.Module):
    """
    CRISPR-GRAM Architecture با قابلیت Ablation Study
    """

    def __init__(
        self,
        input_dim=126,
        embed_dim=128,
        num_nodes=40,
        manifold_dim=96,
        num_layers=4,
        num_heads=8,
        dropout=0.12
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim

        self.embedding = InputEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_nodes=num_nodes,
            dropout=dropout
        )

        self.graph_encoder = GraphEncoder(
            dim=embed_dim,
            layers=num_layers,
            heads=num_heads,
            dropout=dropout
        )

        self.pooling = AttentionPooling(dim=embed_dim)
        self.mean_pooling = MeanPooling()

        self.manifold = ManifoldEncoder(
            input_dim=embed_dim,
            manifold_dim=manifold_dim,
            dropout=dropout
        )

        self.projection = ProjectionHead(
            input_dim=manifold_dim,
            projection_dim=manifold_dim
        )

        # Classifier با ورودی داینامیک (برای حالت‌های مختلف ablation)
        self.classifier = ClassificationHead(
            input_dim=manifold_dim,
            hidden_dim=128,
            dropout=0.3
        )
        
        # Classifier جایگزین برای حالت بدون manifold (ورودی 128)
        self.classifier_no_manifold = ClassificationHead(
            input_dim=embed_dim,
            hidden_dim=128,
            dropout=0.3
        )

        # تنظیمات Ablation (پس از آموزش قابل تغییر هستند)
        self._use_graph = True
        self._use_manifold = True
        self._use_contrastive = True
        self._use_positional = True
        self._use_attention_pooling = True

    def set_ablation_config(self, use_graph=True, use_manifold=True, 
                           use_contrastive=True, use_positional=True, 
                           use_attention_pooling=True):
        """تنظیم پیکربندی Ablation (بدون نیاز به بازآموزی)"""
        self._use_graph = use_graph
        self._use_manifold = use_manifold
        self._use_contrastive = use_contrastive
        self._use_positional = use_positional
        self._use_attention_pooling = use_attention_pooling

    def forward(self, x):
        
        # Embedding
        x = self.embedding(x)
        
        # Positional Embedding (قابل غیرفعال‌سازی)
        if not self._use_positional:
            if hasattr(self.embedding.position, 'position_embedding'):
                x = x - self.embedding.position.position_embedding.unsqueeze(0)
        
        # Graph Encoder (قابل غیرفعال‌سازی)
        if self._use_graph:
            x = self.graph_encoder(x)
        
        # Pooling (قابل غیرفعال‌سازی توجه)
        if self._use_attention_pooling:
            graph_embedding = self.pooling(x)
        else:
            graph_embedding = self.mean_pooling(x)
        
        # Manifold Encoder (قابل غیرفعال‌سازی)
        if self._use_manifold:
            manifold_embedding = self.manifold(graph_embedding)
            logits, probability = self.classifier(manifold_embedding)
        else:
            manifold_embedding = graph_embedding
            logits, probability = self.classifier_no_manifold(manifold_embedding)
        
        # Projection (قابل غیرفعال‌سازی)
        if self._use_contrastive:
            if self._use_manifold:
                projection = self.projection(manifold_embedding)
            else:
                projection = F.normalize(manifold_embedding, p=2, dim=1)
        else:
            projection = torch.zeros_like(manifold_embedding)

        return {
            "logits": logits,
            "probability": probability,
            "graph_embedding": graph_embedding,
            "manifold_embedding": manifold_embedding,
            "projection": projection
        }


if __name__ == "__main__":
    model = CRISPR_GRAM()
    x = torch.randn(8, 126)
    out = model(x)
    print("Full model:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
    
    model.set_ablation_config(use_manifold=False)
    out = model(x)
    print("\nWithout manifold:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
