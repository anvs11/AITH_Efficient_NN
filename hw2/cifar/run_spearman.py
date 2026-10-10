"""Эксперимент: Spearman correlation между ранжированиями разных proxy моделей."""
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from scipy.stats import spearmanr

from datasets import get_cifar_dataloaders
from models import preact20, preact56, preact110
from selection import ForgettingEventsTracker
from train import train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

PROXY_50 = [(1, 0.01), (45, 0.1), (4, 0.01)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 100 if args.dataset == 'cifar100' else 10
    train_loader, test_loader = get_cifar_dataloaders(args.dataset, args.batch_size, args.workers)
    n = len(train_loader.dataset)

    proxy_models = {
        'preact20': preact20(num_classes=num_classes),
        'preact56': preact56(num_classes=num_classes),
        'preact110': preact110(num_classes=num_classes),
    }

    scores_dict = {}

    for name, model in proxy_models.items():
        model = model.to(device)
        tracker = ForgettingEventsTracker(n)

        t0 = time.perf_counter()
        train_model(model, train_loader, test_loader, PROXY_50, device,
                    tracker=tracker, tag=name)
        train_time = time.perf_counter() - t0

        scores = tracker.final_scores()
        scores_dict[name] = scores

        print(f"{name}: обучен за {train_time:.1f}с, inf-примеров: {np.isinf(scores).sum()}")

    print(f"\n=== Spearman Correlation Matrix ===")
    model_names = list(scores_dict.keys())
    correlation_matrix = np.zeros((len(model_names), len(model_names)))

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                s1 = np.where(np.isinf(scores_dict[name1]), 1e10, scores_dict[name1])
                s2 = np.where(np.isinf(scores_dict[name2]), 1e10, scores_dict[name2])
                corr, _ = spearmanr(s1, s2)
                correlation_matrix[i, j] = corr
                print(f"{name1} vs {name2}: {corr:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        'dataset': args.dataset,
        'seed': args.seed,
        'correlation_matrix': correlation_matrix.tolist(),
        'model_names': model_names,
    }
    stamp = time.strftime('%Y%m%d_%H%M%S')
    with open(RESULTS_DIR / f'spearman_{args.dataset}_{stamp}.json', 'w') as f:
        json.dump(record, f, indent=2)

    print(f"\nРезультаты сохранены в {RESULTS_DIR / f'spearman_{args.dataset}_{stamp}.json'}")


if __name__ == '__main__':
    main()
