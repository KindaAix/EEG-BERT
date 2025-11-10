from torch import nn
import torch

class BERTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_dim, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x[:, :, 0])

        word_embeddings = self.embedding(x)
        position_embeddings = self.position_embedding(position_ids)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings