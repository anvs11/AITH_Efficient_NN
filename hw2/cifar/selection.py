"""Методы отбора данных из статьи Selection via Proxy (Coleman et al., ICLR 2020)

Реализованы три критерия core-set selection:
  1. forgetting_events - Алгоритм 2 статьи (Toneva et al., 2019): считаем переходы
     "верно -> неверно" по ходу обучения proxy; примеры, которые модель ни разу
     не классифицировала верно, получают score = inf;
  2. entropy - неопределённость предсказания (выше энтропия = ценнее пример);
  3. grad_norm - норма градиента cross-entropy по последнему linear-слою
     считается аналитически, без backprop.

Плюс random-базовая линия для сравнения.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys

from pathlib import Path
from models import preact20
from datasets import get_cifar_dataloaders

class ForgettingEventsTracker:
    """Счётчик forgetting-событий для каждого примера во время обучения proxy"""

    def __init__(self, n_samples: int):
        self.nevents = np.zeros(n_samples, dtype=np.int64)   # счётчик забываний
        self.last_correct = np.zeros(n_samples, dtype=bool)  # исход прошлой проверки
        self.ever_correct = np.zeros(n_samples, dtype=bool)  # был ли верным хоть раз

    def update(self, indices: np.ndarray, correct: np.ndarray):
        forgotten = self.last_correct[indices] & ~correct
        self.nevents[indices] += forgotten
        self.last_correct[indices] = correct
        self.ever_correct[indices] |= correct

    def final_scores(self) -> np.ndarray:
        """Примеры, которые модель ни разу не угадала, -> inf (как в статье)"""
        scores = self.nevents.astype(np.float64)
        scores[~self.ever_correct] = np.inf
        return scores


@torch.no_grad()
def score_entropy(model, loader, device, n_samples: int) -> np.ndarray:
    """Энтропия предсказаний proxy для каждого примера"""
    model.eval()
    scores = np.zeros(n_samples, dtype=np.float64)
    for x, y, idx in loader:
        p = torch.softmax(model(x.to(device)), dim=1)
        ent = -(p * torch.log(p + 1e-12)).sum(dim=1)
        scores[idx.numpy()] = ent.cpu().numpy()
    return scores


@torch.no_grad()
def score_grad_norm(model, loader, device, n_samples: int) -> np.ndarray:
    """
    Норма градиента CE по весам последнего linear-слоя
    """
    model.eval()
    cache = {}

    def hook(module, inputs, output):
        cache['features'] = inputs[0]

    handle = model.linear.register_forward_hook(hook)
    scores = np.zeros(n_samples, dtype=np.float64)
    try:
        for x, y, idx in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            f = cache['features']
            p = torch.softmax(logits, dim=1)
            diff = p - F.one_hot(y, num_classes=logits.size(1)).to(p.dtype)
            g = diff.norm(dim=1) * f.norm(dim=1)
            scores[idx.cpu().numpy()] = g.cpu().numpy()
    finally:
        handle.remove()
    return scores


def select_top_k(scores: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Индексы k примеров с наибольшим скором"""
    rng = np.random.default_rng(seed)
    tie_noise = rng.random(scores.shape) * 1e-12
    order = np.argsort(-(scores + tie_noise))
    return order[:k]


def select_random(n_samples: int, k: int, seed: int = 42) -> np.ndarray:
    """Random-базовая линия"""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_samples)[:k]


if __name__ == '__main__':
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _ = get_cifar_dataloaders('cifar10', batch_size=128, num_workers=2)
    n = len(train_loader.dataset)
    model = preact20(10).to(device)

    tracker = ForgettingEventsTracker(n)
    with torch.no_grad():
        for x, y, idx in train_loader:
            out = model(x.to(device))
            correct = (out.argmax(1) == y.to(device)).cpu().numpy()
            tracker.update(idx.numpy(), correct)
    fe = tracker.final_scores()

    ent = score_entropy(model, train_loader, device, n)
    gn = score_grad_norm(model, train_loader, device, n)
    sel_ent = select_top_k(ent, 1000)
    sel_rnd = select_random(n, 1000)

    print(f'Всего примеров: {n:,}')
    print(f'forgetting events: inf-примеров = {np.isinf(fe).sum():,}, '
          f'конечных max = {fe[np.isfinite(fe)].max()}')
    print(f'entropy: mean = {ent.mean():.4f}, порог топ-1000 = {ent[sel_ent].min():.4f}')
    print(f'grad_norm: mean = {gn.mean():.4f}, max = {gn.max():.4f}')
    print(f'пример отбора entropy: {sel_ent[:5]}')
    print(f'пример отбора random:  {sel_rnd[:5]}')
