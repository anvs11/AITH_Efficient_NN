"""Эксперимент: влияние количества эпох proxy на качество отбора (CIFAR-100)."""
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from datasets import get_cifar_dataloaders
from models import preact20, wideresnet28_10
from selection import ForgettingEventsTracker, select_top_k, select_random
from train import train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

TARGET_SCHEDULES = {
    200: [(60, 0.1), (60, 0.02), (40, 0.004), (40, 0.0008)],
    100: [(30, 0.1), (30, 0.02), (20, 0.004), (20, 0.0008)],
}


def get_proxy_schedule(n_epochs):
    """Генерирует расписание LR для proxy на n_epochs эпох"""
    if n_epochs == 25:
        return [(1, 0.01), (20, 0.1), (4, 0.01)]
    elif n_epochs == 50:
        return [(1, 0.01), (45, 0.1), (4, 0.01)]
    elif n_epochs == 100:
        return [(1, 0.01), (90, 0.1), (9, 0.01)]
    else:
        raise ValueError(f"Unsupported proxy epochs: {n_epochs}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--proxy-epochs', type=int, default=50,
                   choices=[25, 50, 100],
                   help='Количество эпох обучения proxy')
    p.add_argument('--fraction', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--target-epochs', type=int, default=100, choices=[100, 200])
    p.add_argument('--mode', default='svp', choices=['svp', 'random'])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 100 if args.dataset == 'cifar100' else 10
    train_loader, test_loader = get_cifar_dataloaders(args.dataset, args.batch_size, args.workers)
    n = len(train_loader.dataset)
    timings = {}

    if args.mode == 'svp':
        proxy_schedule = get_proxy_schedule(args.proxy_epochs)
        proxy = preact20(num_classes=num_classes).to(device)
        tracker = ForgettingEventsTracker(n)

        print(f"\nОбучение Proxy ({args.proxy_epochs} эпох)")
        t0 = time.perf_counter()
        train_model(proxy, train_loader, test_loader, proxy_schedule, device,
                    tracker=tracker, tag=f'proxy_{args.proxy_epochs}ep')
        timings['proxy_sec'] = time.perf_counter() - t0

        print(f"\nОтбор данных ({args.mode})")
        t0 = time.perf_counter()
        scores = tracker.final_scores()
        k = int(round(args.fraction * n))
        selected = select_top_k(scores, k, seed=args.seed)
        timings['selection_sec'] = time.perf_counter() - t0
    else:
        print(f"\nСлучайный отбор данных")
        k = int(round(args.fraction * n))
        selected = select_random(n, k, seed=args.seed)
        timings['proxy_sec'] = 0.0
        timings['selection_sec'] = 0.0

    selected = np.sort(selected)
    print(f"Отобрано {k} примеров из {n}")

    target_schedule = TARGET_SCHEDULES[args.target_epochs]
    print(f"\nОбучение Target ({args.target_epochs} эпох)")
    subset_loader = DataLoader(Subset(train_loader.dataset, selected.tolist()),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    target = wideresnet28_10(num_classes=num_classes).to(device)
    t0 = time.perf_counter()
    res = train_model(target, subset_loader, test_loader, target_schedule, device, tag='target')
    timings['target_sec'] = time.perf_counter() - t0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        'dataset': args.dataset,
        'method': args.mode,
        'proxy_epochs': args.proxy_epochs if args.mode == 'svp' else None,
        'target_epochs': args.target_epochs,
        'fraction': args.fraction,
        'seed': args.seed,
        'n_train': n,
        'k': k,
        'target_test_acc': res['test_acc'],
        **timings
    }
    stamp = time.strftime('%Y%m%d_%H%M%S')
    suffix = f'{args.dataset}_{args.mode}_proxy{args.proxy_epochs if args.mode == "svp" else "none"}ep_target{args.target_epochs}ep_{int(args.fraction * 100)}_{stamp}'
    with open(RESULTS_DIR / f'partial_{suffix}.json', 'w') as f:
        json.dump(record, f, indent=2)

    print('\nИтог:')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
