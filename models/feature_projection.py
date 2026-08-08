import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """
    126-D -> Linear -> LayerNorm -> ReLU -> Dropout -> 128-D
    """

    def __init__(self, input_dim=126, embed_dim=128, dropout=0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(x)


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable positional embedding for the virtual graph nodes.
    """

    def __init__(self, num_nodes=40, embed_dim=128):
        super().__init__()

        self.position_embedding = nn.Parameter(
            torch.randn(num_nodes, embed_dim) * 0.02
        )

    def forward(self, x):
        """
        x : (B, num_nodes, embed_dim)
        """
        return x + self.position_embedding.unsqueeze(0)


class RepeatToGraph(nn.Module):
    """
    Replicate one latent vector to `num_nodes` graph nodes.
    """

    def __init__(self, num_nodes=40):
        super().__init__()
        self.num_nodes = num_nodes

    def forward(self, x):
        """
        x : (B, D) -> (B, num_nodes, D)
        """
        return x.unsqueeze(1).repeat(1, self.num_nodes, 1)


class InputEmbedding(nn.Module):
    """
    126 -> Projection -> Repeat -> Positional Embedding -> (B, num_nodes, embed_dim)
    """

    def __init__(self, input_dim=126, embed_dim=128, num_nodes=40, dropout=0.1):
        super().__init__()

        self.project = FeatureProjection(input_dim, embed_dim, dropout)
        self.repeat = RepeatToGraph(num_nodes)
        self.position = LearnablePositionalEmbedding(num_nodes, embed_dim)

    def forward(self, x):
        x = self.project(x)
        x = self.repeat(x)
        x = self.position(x)
        return x


if __name__ == "__main__":
    model = InputEmbedding()
    x = torch.randn(8, 126)
    y = model(x)
    print(y.shape)
