import time
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device, tracker=None):
    """Одна эпоха обучения по алгоритму 2 из статьи"""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        if tracker is not None:
            with torch.no_grad():
                pred_correct = (logits.argmax(1) == y).cpu().numpy()
            tracker.update(idx.numpy(), pred_correct)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Top-1 accuracy на тесте"""
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        n += x.size(0)
    return correct / n


def train_model(model, train_loader, test_loader, segments, device,
                tracker=None, tag='model', log_every=5):
    """Возвращает словарь с test_acc, временем и историей по эпохам"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=segments[0][1],
                                momentum=0.9, weight_decay=5e-4)
    history = []
    epoch = 0
    t0 = time.perf_counter()
    for n_epochs, lr in segments:
        for g in optimizer.param_groups:
            g['lr'] = lr
        for _ in range(n_epochs):
            epoch += 1
            t_ep = time.perf_counter()
            loss, acc = train_one_epoch(model, train_loader, optimizer,
                                        criterion, device, tracker)
            dt = time.perf_counter() - t_ep
            history.append({'epoch': epoch, 'lr': lr, 'loss': loss,
                            'train_acc': acc, 'epoch_sec': dt})
            if epoch % log_every == 0 or epoch == 1:
                print(f'[{tag}] epoch {epoch:3d} | lr {lr:.4f} | '
                      f'loss {loss:.4f} | train_acc {acc:.4f} | {dt:.1f}s')
    total_sec = time.perf_counter() - t0
    test_acc = evaluate(model, test_loader, device)
    print(f'[{tag}] готово: {epoch} эпох за {total_sec:.1f}s | test_acc {test_acc:.4f}')
    return {'test_acc': test_acc, 'train_sec': total_sec, 'history': history}
