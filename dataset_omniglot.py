from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import random
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
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    return loader


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

# omniglot_folders()
