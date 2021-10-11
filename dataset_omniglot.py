from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms


def omniglot_folders():
    omniglot_path = '../omniglot/python/omniglot_resized/'

    characters = []

    for family in os.listdir(omniglot_path):
        if os.path.isdir(os.path.join(omniglot_path, family)):
            for character in os.listdir(os.path.join(omniglot_path, family)):
                characters.append(os.path.join(omniglot_path, family, character).replace('\\', '/'))

    # print(characters)

    random.seed(1)
    random.shuffle(characters)

    # print(len(characters))

    train_num = 1200

    characters_train = characters[:train_num]
    characters_val = characters[train_num:]

    return characters_train, characters_val


def get_data_loader(task, num_per_class=1, split='train', shuffle=True, rotation=0):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    dataset = Omniglot(task, split=split,
                       transform=transforms.Compose([Rotate(rotation), transforms.ToTensor(), normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.num_train, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.num_test, shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    return loader


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_instances' examples each from 'num_classes' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_classes, num_instances, shuffle=True):
        self.num_per_class = num_per_class
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_instances for i in torch.randperm(self.num_instances)[:self.num_per_class]] for j
                     in
                     range(self.num_classes)]
        else:
            batch = [[i + j * self.num_instances for i in range(self.num_instances)[:self.num_per_class]] for j in
                     range(self.num_classes)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        image = image.resize((28, 28), resample=Image.LANCZOS)
        # image = np.array(image, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class OmniglotTask(object):
    def __init__(self, characters, num_classes, num_train, num_test):
        self.characters = characters
        self.num_classes = num_classes
        self.num_train = num_train
        self.num_test = num_test

        # choose num_classes (5 here) images from characters (a bunch of character images)
        batch = random.sample(self.characters, self.num_classes)
        labels = np.array(range(len(batch)))
        labels = dict(zip(batch, labels))
        samples = dict()

        print(labels)

        self.train_roots = []
        self.test_roots = []

        for e in batch:
            temp = []
            for x in os.listdir(e):
                temp.append(os.path.join(e, x))
            # print(temp)

            samples[e] = random.sample(temp, len(temp))

            # print(samples)

            self.train_roots += samples[e][:num_train]  # 5
            self.test_roots += samples[e][num_train:num_train + num_test]  # 5:5+15

            # print(self.train_roots)
            # print(self.test_roots)

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

        # print(self.train_labels)

    def get_class(self, sample):
        return os.path.join(sample.split('\\')[0])


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x, mode='reflect'):
        x = x.rotate(self.angle)
        return
