from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset_omniglot


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),  # in_channel 1 or 3?
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RelationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # why is in_channel 128? because after feature concatenation.
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out


def weights_init(m):
    class_name = m.__class__.__name__


def main():
    episode = 1000000
    test_episode = 1000
    feature_dim = 64
    GPU = 0
    learning_rate = 0.001
    num_class = 5
    batch_num_per_class = 15
    sample_num_per_class = 5

    cnn_encoder = CNNEncoder()
    relation_network = RelationNetwork(64, 8)

    cnn_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    cnn_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    characters_train, characters_val = dataset_omniglot.omniglot_folders()

    # print(cnn_encoder.parameters(), relation_network.parameters())

    cnn_encoder_optim = torch.optim.Adam(cnn_encoder.parameters(), lr=learning_rate)
    cnn_encoder_scheduler = StepLR(cnn_encoder_optim, step_size=100000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=learning_rate)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    print('Training...')

    last_accuracy = 0.0

    for e in range(episode):
        cnn_encoder_scheduler.step(e)
        relation_network_scheduler.step(e)

        degrees = random.choice([0, 90, 180, 270])
        task = dataset_omniglot.OmniglotTask(characters_train, num_class, sample_num_per_class, batch_num_per_class)
        train_dataloader = dataset_omniglot.get_data_loader(task, num_per_class=sample_num_per_class, split='train',
                                                            shuffle=False, rotation=degrees)
        test_dataloader = dataset_omniglot.get_data_loader(task, num_per_class=sample_num_per_class, split='test',
                                                           shuffle=True, rotation=degrees)

        train_images, train_labels = train_dataloader.__iter__().next()
        test_images, test_labels = test_dataloader.__iter__().next()

        train_features = cnn_encoder(Variable(train_images).cuda(GPU))
        train_features = train_features.view(num_class, sample_num_per_class, feature_dim, 5, 5)
        train_features = torch.sum(train_features, 1).squeeze(1)
        test_features = cnn_encoder(Variable(test_images).cuda(GPU))

        # calculate relations
        train_features_ext = train_features.unsqueeze(0).repeat(batch_num_per_class * num_class, 1, 1, 1, 1)
        test_features_ext = test_features.unsqueeze(0).repeat(num_class, 1, 1, 1, 1)
        test_features_ext = torch.transpose(test_features_ext, 0, 1)

        relation_pair = torch.cat((train_features_ext, test_features_ext), 2).view(-1, feature_dim * 2, 5, 5)
        relations = relation_network(relation_pair).view(-1, num_class)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(
            torch.zeros(batch_num_per_class * num_class, num_class).scatter_(1, test_labels.view(-1, 1), 1)).cuda(GPU)
        loss = mse(relations, one_hot_labels)

        # training
        cnn_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(cnn_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        cnn_encoder_optim.step()
        relation_network_optim.step()

        if (e + 1) % 100 == 0:
            print('e: ', e + 1, 'loss: ', loss.data[0])
        # testing
        if (e + 1) % 5000 == 0:
            print('Testing...')
            total_rewards = 0

            for i in range(test_episode):
                degrees = random.choice([0, 90, 180, 270])
                task = dataset_omniglot.OmniglotTask(characters_val, num_class, sample_num_per_class,
                                                     sample_num_per_class, )
                train_dataloader = dataset_omniglot.get_data_loader(task, num_per_class=sample_num_per_class,
                                                                    split='train',
                                                                    shuffle=False, rotation=degrees)
                test_dataloader = dataset_omniglot.get_data_loader(task, num_per_class=sample_num_per_class,
                                                                   split='test',
                                                                   shuffle=True, rotation=degrees)

                train_images, train_labels = train_dataloader.__iter__().next()
                test_images, test_labels = test_dataloader.__iter__().next()

                # calculate features
                train_features = cnn_encoder(Variable(train_images)).cuda(GPU)
                train_features = train_features.view(num_class, sample_num_per_class, feature_dim, 5, 5)
                train_features = torch.sum(train_features, 1).squeeze(1)
                test_features = cnn_encoder(Variable(test_images).cuda(GPU).cuda(GPU))

                # calculate relations
                train_features_ext = train_features.unsqueeze(0).repeat(sample_num_per_class * num_class, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(num_class, 1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)

                relation_pair = torch.cat((train_features_ext, test_features_ext), 2).view(-1, feature_dim * 2, 5, 5)
                relations = relation_network(relation_pair).view(-1, num_class)

                _, predict_labels = torch.max(relations.data, 1)

                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in
                           range(num_class * sample_num_per_class)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards / 1.0 / num_class / sample_num_per_class / test_episode

            print('test_accuracy: ', test_accuracy)

            if test_accuracy > last_accuracy:
                torch.save(cnn_encoder.state_dict(),
                           str('./models/omniglot_cnn_encoder_' + str(num_class) + 'way_' + str(
                               sample_num_per_class) + 'shot.pkl'))
                torch.save(relation_network.state_dict(),
                           str('./models/omniglot_relation_network_' + str(num_class) + 'way_' + str(
                               sample_num_per_class) + 'shot.pkl'))

                print('Save network for episode: ', e)

                last_accuracy = test_accuracy


if __name__ == '__main__':
    main()
