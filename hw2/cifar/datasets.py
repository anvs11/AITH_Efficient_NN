import torch
from torchvision import datasets, transforms
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / 'data'


class IndexedDataset(torch.utils.data.Dataset):
    """Обёртка: возвращает (image, label, index)
    Индекс нужен для forgetting events и для отбора подмножеств"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return x, y, i


def get_cifar_dataloaders(dataset_name='cifar10', batch_size=128, num_workers=4):
    """
    Возвращает train и test DataLoader для CIFAR
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=str(DATA_ROOT), train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=str(DATA_ROOT), train=False, download=True, transform=transform_test)
    else:
        raise ValueError("Unsupported dataset")

    train_dataset = IndexedDataset(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
