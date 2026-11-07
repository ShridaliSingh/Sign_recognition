import torch
import torch.nn as nn


class ConvBaseline(nn.Module):
    def __init__(self, in_dim=320, hidden=256, n_classes=250, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, mask):
            m = mask.unsqueeze(-1).float()
            h = torch.relu(self.proj(x)) * m
            h = h.transpose(1, 2)
            h = torch.relu(self.conv1(h)) * m.transpose(1, 2)
            h = torch.relu(self.conv2(h)) * m.transpose(1, 2)
            h = h.transpose(1, 2)
            pooled = h.sum(dim=1) / m.sum(dim=1).clamp(min=1)
            return self.head(self.dropout(pooled))

class SignTransformer(nn.Module):
    def __init__(self, in_dim=320, d_model=192, n_heads=4, n_layers=4,
                 n_classes=250, max_len=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model))
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, mask):
        T = x.shape[1]
        h = self.proj(x) + self.pos[:, :T]
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.norm(h)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.head(self.dropout(pooled))

    