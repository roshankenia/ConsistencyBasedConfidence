import numpy as np
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100
from .vec_cifar import VECCIFAR10
import os
import sys
import torch
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


def vec_input_dataset(dataset, noise_type, noise_path, is_human):
    if dataset == 'cifar10':
        train_dataset = VECCIFAR10(root='~/data/',
                                   download=True,
                                   train=True,
                                   noise_type=noise_type,
                                   noise_path=noise_path, is_human=is_human
                                   )
        test_dataset = VECCIFAR10(root='~/data/',
                                   download=False,
                                   train=False,
                                   noise_type=noise_type
                                   )
        num_classes = 10
        num_training_samples = 50000
    return train_dataset, test_dataset, num_classes, num_training_samples


def input_dataset(dataset, noise_type, noise_path, is_human):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root='~/data/',
                                download=True,
                                train=True,
                                transform=train_cifar10_transform,
                                noise_type=noise_type,
                                noise_path=noise_path, is_human=is_human
                                )
        test_dataset = CIFAR10(root='~/data/',
                               download=False,
                               train=False,
                               transform=test_cifar10_transform,
                               noise_type=noise_type
                               )
        num_classes = 10
        num_training_samples = 50000
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root='~/data/',
                                 download=True,
                                 train=True,
                                 transform=train_cifar100_transform,
                                 noise_type=noise_type,
                                 noise_path=noise_path, is_human=is_human
                                 )
        test_dataset = CIFAR100(root='~/data/',
                                download=False,
                                train=False,
                                transform=test_cifar100_transform,
                                noise_type=noise_type
                                )
        num_classes = 100
        num_training_samples = 50000
    return train_dataset, test_dataset, num_classes, num_training_samples
