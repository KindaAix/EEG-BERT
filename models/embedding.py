from torch import nn
import torch

class WhisperEmbedding(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(in_channels=config.n_features, out_channels=config.embedding_size, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=config.embedding_size, out_channels=config.embedding_size, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x : Tensor of shape (batch_size, seq_len, n_features)
        """
        x = self.gelu(self.conv1(x.permute(0, 2, 1)))  # Change to (batch_size, n_features, seq_len)
        embeddings = self.gelu(self.conv2(x))
        return embeddings.permute(0, 2, 1)  # Back to (batch_size, new_seq_len, embedding_size)
    

if __name__ == "__main__":
    class Config:
        n_features = 744
        embedding_size = 1024
    
    config = Config()
    model = WhisperEmbedding(config).to("cuda")
    sample_input = torch.randn(1, 160, config.n_features).to("cuda")  # (batch_size, seq_len, n_features)
    output = model(sample_input)
    print(output.shape)