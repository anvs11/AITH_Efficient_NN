import argparse
import json
import time
import numpy as np
import torch

from models import preact20, wideresnet28_10
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from datasets import get_cifar_dataloaders
from selection import (ForgettingEventsTracker, score_entropy,
                             score_grad_norm, select_top_k, select_random)
from train import train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

PROXY_50 = [(1, 0.01), (45, 0.1), (4, 0.01)]
TARGET_FULL = [(60, 0.1), (60, 0.02), (40, 0.004), (40, 0.0008)]
TARGET_QUICK = [(20, 0.1), (10, 0.02)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--method', default='forgetting_events',
                   choices=['forgetting_events', 'entropy', 'grad_norm', 'random'])
    p.add_argument('--fraction', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true', help='короткое расписание target для отладки')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--proxy-arch', default='preact20', choices=['preact20', 'wideresnet28_10'])
    p.add_argument('--target-arch', default='wideresnet28_10', choices=['preact20', 'wideresnet28_10'])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 10 if args.dataset == 'cifar10' else 100
    train_loader, test_loader = get_cifar_dataloaders(args.dataset, args.batch_size, args.workers)
    score_loader = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    n = len(train_loader.dataset)
    timings = {}

    model_factory = {
        'preact20': preact20,
        'wideresnet28_10': wideresnet28_10
    }

    if args.method == 'random':
        scores, timings['proxy_sec'] = None, 0.0
    else:
        proxy = model_factory[args.proxy_arch](num_classes=num_classes).to(device)
        tracker = ForgettingEventsTracker(n) if args.method == 'forgetting_events' else None
        t0 = time.perf_counter()
        train_model(proxy, train_loader, test_loader, PROXY_50, device,
                    tracker=tracker, tag='proxy')
        timings['proxy_sec'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if args.method == 'forgetting_events':
            scores = tracker.final_scores()
        elif args.method == 'entropy':
            scores = score_entropy(proxy, score_loader, device, n)
        else:
            scores = score_grad_norm(proxy, score_loader, device, n)
        timings['selection_sec'] = time.perf_counter() - t0

    k = int(round(args.fraction * n))
    selected = (select_random(n, k, seed=args.seed) if scores is None
                else select_top_k(scores, k, seed=args.seed))
    selected = np.sort(selected)

    subset_loader = DataLoader(Subset(train_loader.dataset, selected.tolist()),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    target = model_factory[args.target_arch](num_classes=num_classes).to(device)
    t0 = time.perf_counter()
    res = train_model(target, subset_loader, test_loader,
                      TARGET_QUICK if args.quick else TARGET_FULL, device, tag='target')
    timings['target_sec'] = time.perf_counter() - t0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {'dataset': args.dataset, 'method': args.method,
              'fraction': args.fraction, 'seed': args.seed,
              'n_train': n, 'k': k, 'target_test_acc': res['test_acc'], **timings}
    stamp = time.strftime('%Y%m%d_%H%M%S')
    suffix = f'{args.dataset}_{args.method}_{int(args.fraction * 100)}_{stamp}'
    with open(RESULTS_DIR / f'coreset_{suffix}.json', 'w') as f:
        json.dump(record, f, indent=2)
    np.savetxt(RESULTS_DIR / f'selected_{suffix}.index', selected, fmt='%d')

    print('\nИтог:')
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
