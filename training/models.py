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