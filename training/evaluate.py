from collections import Counter
import torch


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, mask, y in loader:
        all_logits.append(model(x.to(device), mask.to(device)).cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def topk_accuracy(logits, labels, k):
    topk = logits.topk(k, dim=1).indices
    return (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()


def confused_pairs(logits, labels, idx_to_sign, n=10):
    preds = logits.argmax(dim=1)
    wrong = preds != labels
    pairs = Counter(zip(labels[wrong].tolist(), preds[wrong].tolist()))
    return [(idx_to_sign[t], idx_to_sign[p], c) for (t, p), c in pairs.most_common(n)]