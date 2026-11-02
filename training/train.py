import torch
import torch.nn as nn


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            loss = loss_fn(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(y)

    return total_loss / count, correct / count