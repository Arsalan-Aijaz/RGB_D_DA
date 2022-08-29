from __future__ import print_function, division
import os

import torch
import pandas as pd

from NeuralNetwork import ResBase, MainHead, RotationHead
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import copy
import torch.nn as nn
import torch.optim as optim
from itertools import permutations
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
import random

perms = list(permutations(range(4)))
mean = [0.485, 0.456, 0.40]
std = [0.229, 0.224, 0.225]


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image_c, image_d):

        h1, w1 = image_c.height, image_c.width
        h2, w2 = image_d.height, image_d.width
        if isinstance(self.output_size, int):
            if h1 > w1:
                new_h1, new_w1 = self.output_size * h1 / w1, self.output_size
            else:
                new_h1, new_w1 = self.output_size, self.output_size * w1 / h1
            if h2 > w2:
                new_h2, new_w2 = self.output_size * h2 / w2, self.output_size
            else:
                new_h2, new_w2 = self.output_size, self.output_size * w2 / h2
        else:
            new_h1, new_w1 = self.output_size
            new_h2, new_w2 = self.output_size

        new_h1, new_w1 = int(new_h1), int(new_w1)
        new_h2, new_w2 = int(new_h2), int(new_w2)

        img1 = image_c.resize((new_h1, new_w1))
        img2 = image_d.resize((new_h2, new_w2))

        return img1, img2


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image_c, image_d):
        h1, w1 = image_c.height, image_c.width
        h2, w2 = image_d.height, image_d.width

        new_h1, new_w1 = self.output_size
        new_h2, new_w2 = self.output_size

        top1 = np.random.randint(0, h1 - new_h1)
        left1 = np.random.randint(0, w1 - new_w1)
        top2 = np.random.randint(0, h2 - new_h2)
        left2 = np.random.randint(0, w2 - new_w2)

        img1 = image_c.crop((left1, h1 - new_h1, left1 + new_w1, h1))
        img2 = image_d.crop((left2, h2 - new_h2, left2 + new_w2, h2))

        return img1, img2


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image_c, image_d):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = transforms.ToTensor()(image_c)
        img2 = transforms.ToTensor()(image_d)
        return img1, img2


scale = Rescale(256)
crop = RandomCrop(224)
tensor = ToTensor()


def make_jigsaw(img1, z):
    img1 = img1.to('cpu')
    img2 = transforms.ToPILImage()(img1[0:3])
    img2 = img2.resize((224, 224))
    img1 = transforms.ToPILImage()(img1[3:6])
    img1 = img1.resize((224, 224))
    blockmap1 = [(0, 0, 112, 112), (0, 112, 112, 224), (112, 0, 224, 112), (112, 112, 224, 224)]
    blockmap2 = [(0, 0, 112, 112), (0, 112, 112, 224), (112, 0, 224, 112), (112, 112, 224, 224)]
    shuffle = list(blockmap1)
    shuffle = list(map(shuffle.__getitem__, perms[z]))
    result1 = Image.new(mode="RGB", size=(224, 224))
    result2 = Image.new(mode="RGB", size=(224, 224))
    for box1, box2, sbox in zip(blockmap1, blockmap2, shuffle):
        c = img1.crop(sbox)
        d = img2.crop(sbox)
        result1.paste(c, box1)
        result2.paste(d, box2)
    result = torch.cat((transforms.functional.pil_to_tensor(result1), transforms.functional.pil_to_tensor(result2)), 0)
    return result


class CustomImageDataset(Dataset):
    def __init__(self, paths_c, paths_d, img_dir, DoRot=False, labels="",
                 transform=None, ROD=False, Jigsaw=False):
        self.paths_c = paths_c
        self.paths_d = paths_d
        self.img_labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.DoRot = DoRot
        self.Jigsaw = Jigsaw
        self.ROD = ROD

    def __len__(self):
        return len(self.paths_c)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.ROD:
            rgb_path = os.path.join(self.img_dir, "rgb-washington/", self.paths_c.values[idx])
            depth_path = os.path.join(self.img_dir, "surfnorm-washington/", self.paths_d.values[idx])
            image_c = Image.open(rgb_path)
            image_d = Image.open(depth_path)
        else:
            rgb_path = os.path.join(self.img_dir, self.paths_c[idx])
            depth_path = os.path.join(self.img_dir, self.paths_d[idx])
            image_c = Image.open(rgb_path)
            image_d = Image.open(depth_path)
        if self.DoRot:
            z = random.choice((0, 1, 2, 3))
        elif self.Jigsaw:
            z = len(perms) - 1
            z = random.randint(0, z)
        else:
            z = 0
        if self.img_labels != "" and self.img_labels != []:
            label = self.img_labels[idx]
        else:
            label = 0
        if self.transform:
            image_c, image_d = scale(image_c, image_d)
            image_c, image_d = crop(image_c, image_d)
            image_c, image_d = tensor(image_c, image_d)
            transforms.Normalize(mean, std, inplace=True)(image_c)
            transforms.Normalize(mean, std, inplace=True)(image_d)
        x = torch.cat((image_c, image_d), 0)
        return x, label, z


def prep_data(type, batch_size):
    # prepare data
    df_synROD_train = pd.read_csv('newdata/ROD-synROD/synROD/synARID_50k-split_sync_train1.txt',
                                  delimiter=' ', header=0, names=['File', 'Label'])
    df_synROD_test = pd.read_csv('newdata/ROD-synROD/synROD/synARID_50k-split_sync_test1.txt',
                                 delimiter=' ', header=0, names=['File', 'Label'])
    df_ROD_train = pd.read_csv('newdata/ROD-synROD/ROD/wrgbd_40k-split_sync.txt',
                               delimiter=' ', header=0, names=['File', 'Label'])

    train_synROD = dict(RGBfile=[], Dfile=[], Label=[])
    test_synROD = dict(RGBfile=[], Dfile=[], Label=[])
    train_ROD = dict(RGBfile=[], Dfile=[], Label=[])
    test_ROD = dict(RGBfile=[], Dfile=[])

    train_synROD["RGBfile"] = (df_synROD_train["File"].apply(lambda x: x.replace("***", "rgb"))).map(str)
    train_synROD["Dfile"] = (df_synROD_train["File"].apply(lambda x: x.replace("***", "depth"))).map(str)
    train_synROD["Label"] = df_synROD_train["Label"].values.tolist()

    test_synROD["RGBfile"] = (df_synROD_test["File"].apply(lambda x: x.replace("***", "rgb"))).map(str)
    test_synROD["Dfile"] = (df_synROD_test["File"].apply(lambda x: x.replace("***", "depth"))).map(str)
    test_synROD["Label"] = df_synROD_test["Label"].values.tolist()

    train_ROD["RGBfile"] = (df_ROD_train["File"].apply(lambda x: x.replace("***", "crop"))).map(str)
    train_ROD["Dfile"] = (df_ROD_train["File"].apply(lambda x: x.replace("***", "depthcrop"))).map(str)

    test_ROD["RGBfile"] = (df_ROD_train["File"].apply(lambda x: x.replace("***", "crop"))).map(str)
    test_ROD["Dfile"] = (df_ROD_train["File"].apply(lambda x: x.replace("***", "depthcrop"))).map(str)
    test_ROD["Label"] = df_ROD_train["Label"].values.tolist()

    if type == 'Jigsaw':
        training_data_synROD = CustomImageDataset(train_synROD["RGBfile"], train_synROD["Dfile"],
                                                  img_dir="newdata/ROD-synROD/synROD/",
                                                  labels=train_synROD["Label"],
                                                  transform=True, ROD=False, Jigsaw=True)

        RGB_train, RGB_test, label_train, label_test = train_test_split(test_ROD["RGBfile"], test_ROD["Label"],
                                                                        test_size=0.3, random_state=42)
        D_train, D_test, _, _ = train_test_split(test_ROD["Dfile"], test_ROD["Label"], test_size=0.3, random_state=42)
        test_data = CustomImageDataset(RGB_test, D_test, img_dir="newdata/ROD-synROD/ROD/",
                                       labels=label_test, transform=True, ROD=True, Jigsaw=True)
        training_data_ROD = CustomImageDataset(RGB_train, D_train, img_dir="newdata/ROD-synROD/ROD/",
                                               transform=True, ROD=True, Jigsaw=True)
        train_dataloader_synROD = DataLoader(training_data_synROD, batch_size=batch_size, shuffle=True, pin_memory=True,
                                             num_workers=2, drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
        dataloaders = dict(train=train_dataloader_synROD, val=test_dataloader)
    elif type == 'DA':

        training_data_synROD = CustomImageDataset(train_synROD["RGBfile"], train_synROD["Dfile"], DoRot=True,
                                                  img_dir="newdata/ROD-synROD/synROD/",
                                                  labels=train_synROD["Label"],
                                                  transform=True, ROD=False)

        RGB_train, RGB_test, label_train, label_test = train_test_split(test_ROD["RGBfile"], test_ROD["Label"],
                                                                        test_size=0.3, random_state=42)
        D_train, D_test, _, _ = train_test_split(test_ROD["Dfile"], test_ROD["Label"],
                                                 test_size=0.3, random_state=42)
        test_data = CustomImageDataset(RGB_test, D_test, DoRot=False,
                                       img_dir="newdata/ROD-synROD/ROD/", labels=label_test,
                                       transform=True, ROD=True)
        training_data_ROD = CustomImageDataset(RGB_train, D_train, DoRot=True,
                                               img_dir="newdata/ROD-synROD/ROD/", transform=True, ROD=True)
        train_dataloader_synROD = DataLoader(training_data_synROD, batch_size=batch_size, shuffle=True, pin_memory=True,
                                             num_workers=2, drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2,
                                     drop_last=True)
        dataloaders = dict(train=train_dataloader_synROD, val=test_dataloader)
    else:
        training_data_e2e = CustomImageDataset(train_synROD["RGBfile"], train_synROD["Dfile"], DoRot=False,
                                               img_dir="newdata/ROD-synROD/synROD/", labels=train_synROD["Label"],
                                               transform=True, ROD=False)
        test_data_e2e = CustomImageDataset(test_ROD["RGBfile"], train_ROD["Dfile"], DoRot=False,
                                           img_dir="newdata/ROD-synROD/ROD/", labels=test_ROD["Label"],
                                           transform=True, ROD=True)
        training_data_ROD = CustomImageDataset(train_ROD["RGBfile"], train_ROD["Dfile"], DoRot=True,
                                               img_dir="newdata/ROD-synROD/ROD/", transform=True, ROD=True)
        train_dataloader_e2e = DataLoader(training_data_e2e, batch_size=batch_size, shuffle=True, pin_memory=True,
                                          num_workers=4)
        test_dataloader_e2e = DataLoader(test_data_e2e, batch_size=batch_size, shuffle=True, pin_memory=True,
                                         num_workers=4)
        dataloaders = dict(train=train_dataloader_e2e, val=test_dataloader_e2e)

    train_dataloader_ROD = DataLoader(training_data_ROD, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=0, drop_last=True)

    return dataloaders, train_dataloader_ROD


def train_model_DA(num_epochs=20, batch_size=64, mode="Rotation"):
    FE_rgb = ResBase()
    FE_depth = ResBase()
    FC_M = MainHead(input_dim=512 * 2, class_num=47, dropout_p=0.5, extract=False)
    if mode == "Rotation":
        FC_P = RotationHead(input_dim=1024, class_num=4)
        dataloaders, target_dataloader = prep_data('DA', batch_size)
    elif mode == "Jigsaw":
        FC_P = RotationHead(input_dim=1024, class_num=24)
        dataloaders, target_dataloader = prep_data('Jigsaw', batch_size)
    else:
        FC_P = RotationHead(input_dim=1024, class_num=4)
        dataloaders,_ = prep_data('Source', batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FE_rgb, FE_depth, FC_M, FC_P = FE_rgb.to(device), FE_depth.to(device), FC_M.to(device), FC_P.to(device)
    criterion = nn.CrossEntropyLoss()
    pretext_weight = 0.1
    optimizers = [optim.SGD(FE_rgb.parameters(), lr=3e-4, weight_decay=0.05, momentum=0.9),
                  optim.SGD(FE_depth.parameters(), lr=3e-4, weight_decay=0.05, momentum=0.9),
                  optim.SGD(FC_M.parameters(), lr=3e-4, weight_decay=0.05, momentum=0.9),
                  optim.SGD(FC_P.parameters(), lr=3e-4, weight_decay=0.05, momentum=0.9)]
    since = time.time()
    best_model_wts = copy.deepcopy(FE_rgb.state_dict())
    best_acc = 0.0

    # Store losses and accuracies across epochs
    losses, accuracies, lr = dict(train=[], val=[]), dict(train=[], val=[]), 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                FE_rgb.train()
                FE_depth.train()
                FC_M.train()
                FC_P.train()
            else:
                FE_rgb.eval()
                FE_depth.eval()
                FC_M.eval()
                FC_P.eval()

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            nsamples = 0
            for x_source, y_source, z_source in dataloaders[phase]:  # Loading mini-batch from S
                # Load mini-batch from S
                x_source, y_source = x_source.to(device), y_source.to(device)
                nsamples += x_source.shape[0]
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Compute main loss Lm
                    feats_rgb, _ = FE_rgb(x_source[:, [0, 1, 2]])
                    feats_depth, _ = FE_depth(x_source[:, [3, 4, 5]])
                    feats = torch.cat((feats_rgb, feats_depth), dim=1)
                    pred_logits_cat = FC_M(feats)
                    _, preds_cat = torch.max(pred_logits_cat, 1)
                    Lm = criterion(pred_logits_cat, y_source)

                    # Load mini-batches from S~ and T~
                    x_target, _, z_target = next(iter(target_dataloader))

                    # print_image(x_target[0])
                    # print_image(x_target_tilde[0])
                    x_target = x_target.to(device)

                    InTilde = torch.zeros(size=(batch_size, 6, 224, 224), device=device)
                    InZ = torch.zeros(size=[batch_size], device=device, dtype=torch.uint8)
                    for idx in range(0, batch_size, 2):
                        if mode == "Jigsaw":
                            InTilde[idx] = make_jigsaw(x_source[idx], z_source[idx].item())
                            InTilde[idx + 1] = make_jigsaw(x_target[idx], z_target[idx].item())
                        else:
                            rot_c = random.choice((0, 1, 2, 3))
                            rot_d = z_source[idx].item() + rot_c
                            if rot_d >= 4:
                                rot_d -= 4
                            InTilde[idx][0:3] = transforms.functional.rotate(x_source[idx][0:3], rot_c * 90)
                            InTilde[idx][3:6] = transforms.functional.rotate(x_source[idx][3:6], rot_d * 90)
                            rot_c = random.choice((0, 1, 2, 3))
                            rot_d = z_target[idx].item() + rot_c
                            if rot_d > 4:
                                rot_d -= 4
                            InTilde[idx + 1] = transforms.functional.rotate(x_target[idx], z_target[idx].item() * 90)
                        InZ[idx] = z_source[idx].data
                        InZ[idx + 1] = z_target[idx].data
                    _, feats_rgb = FE_rgb(InTilde[:, [0, 1, 2]])
                    _, feats_depth = FE_depth(InTilde[:, [3, 4, 5]])
                    feats = torch.cat((feats_rgb, feats_depth), dim=1)
                    pred_logits_rot = FC_P(feats)
                    _, preds_rot = torch.max(pred_logits_rot, 1)
                    Lp = criterion(pred_logits_rot, InZ)

                    # Cross-entropy minimization
                    L = Lm + (pretext_weight * Lp)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        L.backward()
                        for optimizer in optimizers:
                            optimizer.step()
                # statistics
                running_loss += Lm.item() * (x_target.size(0))
                running_corrects += (preds_cat == y_source.data).sum().item()
                print('Phase: {} Progress: {}/{} Lm: {:.4f} Lp: {:.4f} L: {:.4f}'.format(phase, nsamples, dataloaders[
                    phase].sampler.num_samples, Lm.item(), Lp.item(), L.item()))

            epoch_loss = running_loss / nsamples
            epoch_acc = running_corrects / nsamples

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            lr = optimizers[0].param_groups[0]['lr']
            print('Phase: {} LR: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, lr, epoch_loss, epoch_acc))
            with open('result_rotation.txt', 'a') as fp:
                fp.write('Phase: {}, LR: {:.4f}, Loss: {:.4f}, Acc: {:.4f}\n'.format(phase, lr, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_FC_M_wts = copy.deepcopy(FC_M.state_dict())
                best_FC_P_wts = copy.deepcopy(FC_P.state_dict())
                best_FE_rgb_wts = copy.deepcopy(FE_rgb.state_dict())
                best_FE_depth_wts = copy.deepcopy(FE_depth.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open('result_rotation.txt', 'a') as fp:
        fp.write(('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)))
        fp.write('Best val Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    FC_M.load_state_dict(best_FC_M_wts)
    FC_P.load_state_dict(best_FC_P_wts)
    FE_rgb.load_state_dict(best_FE_rgb_wts)
    FE_depth.load_state_dict(best_FE_depth_wts)
    return FC_M, FC_P, FE_rgb, FE_depth, losses, accuracies, lr


def train_model_source_only(num_epochs=20, batch_size=64, mode="Source"):
    FE_rgb = ResBase()
    FE_depth = ResBase()
    FC_M = MainHead(input_dim=512 * 2, class_num=47, dropout_p=0.5, extract=False)
    if mode == "Rotation":
        FC_P = RotationHead(input_dim=1024, class_num=4)
        dataloaders, target_dataloader = prep_data('DA', batch_size)
    elif mode == "Jigsaw":
        FC_P = RotationHead(input_dim=1024, class_num=24)
        dataloaders, target_dataloader = prep_data('Jigsaw', batch_size)
    else:
        FC_P = RotationHead(input_dim=1024, class_num=4)
        dataloaders, _ = prep_data('Source', batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FE_rgb, FE_depth, FC_M, FC_P = FE_rgb.to(device), FE_depth.to(device), FC_M.to(device), FC_P.to(device)
    criterion = nn.CrossEntropyLoss()
    pretext_weight = 0.1
    optimizers = [optim.SGD(FE_rgb.parameters(), lr=3e-4, weight_decay=0.05, momentum=0.9),
                  optim.SGD(FE_depth.parameters(), lr=3e-4, weight_decay=0.05, momentum=0.9),
                  optim.SGD(FC_M.parameters(), lr=3e-4, weight_decay=0.05, momentum=0.9)]
    since = time.time()
    best_model_wts = copy.deepcopy(FE_rgb.state_dict())
    best_acc = 0.0

    # Store losses and accuracies across epochs
    losses, accuracies, lr = dict(train=[], val=[]), dict(train=[], val=[]), 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                FE_rgb.train()
                FE_depth.train()
                FC_M.train()
                FC_P.train()
            else:
                FE_rgb.eval()
                FE_depth.eval()
                FC_M.eval()
                FC_P.eval()

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            nsamples = 0
            for x_source, y_source, z_source in dataloaders[phase]:  # Loading mini-batch from S
                # Load mini-batch from S
                x_source, y_source = x_source.to(device), y_source.to(device)
                nsamples += x_source.shape[0]
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Compute main loss Lm
                    feats_rgb, _ = FE_rgb(x_source[:, [0, 1, 2]])
                    feats_depth, _ = FE_depth(x_source[:, [3, 4, 5]])
                    feats = torch.cat((feats_rgb, feats_depth), dim=1)
                    pred_logits_cat = FC_M(feats)
                    _, preds_cat = torch.max(pred_logits_cat, 1)
                    Lm = criterion(pred_logits_cat, y_source)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        Lm.backward()
                        for optimizer in optimizers:
                            optimizer.step()
                # statistics
                running_loss += Lm.item() * (x_source.size(0))
                running_corrects += (preds_cat == y_source.data).sum().item()
                # print('Phase: {} Progress: {}/{} Lm: {:.4f} Lp: {:.4f} L: {:.4f}'.format(phase, nsamples, dataloaders[
                #    phase].sampler.num_samples, Lm.item(), Lp.item(), L.item()))

            epoch_loss = running_loss / nsamples
            epoch_acc = running_corrects / nsamples

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            lr = optimizers[0].param_groups[0]['lr']
            print('Phase: {} LR: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, lr, epoch_loss, epoch_acc))
            with open('result_source.txt', 'a') as fp:
                fp.write('Phase: {}, LR: {:.4f}, Loss: {:.4f}, Acc: {:.4f}\n'.format(phase, lr, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_FC_M_wts = copy.deepcopy(FC_M.state_dict())
                best_FC_P_wts = copy.deepcopy(FC_P.state_dict())
                best_FE_rgb_wts = copy.deepcopy(FE_rgb.state_dict())
                best_FE_depth_wts = copy.deepcopy(FE_depth.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open('result_source.txt', 'a') as fp:
        fp.write(('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)))
        fp.write('Best val Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    FC_M.load_state_dict(best_FC_M_wts)
    FC_P.load_state_dict(best_FC_P_wts)
    FE_rgb.load_state_dict(best_FE_rgb_wts)
    FE_depth.load_state_dict(best_FE_depth_wts)
    return FC_M, FC_P, FE_rgb, FE_depth, losses, accuracies, lr

if __name__ == '__main__':
    _, _, _, _, losses, accuracies, lr = train_model_source_only(num_epochs=1, batch_size=32, mode="Source")
    _, _, _, _, losses, accuracies, lr = train_model_DA(num_epochs=1, batch_size=32, mode="Rotation")
    _, _, _, _, losses, accuracies, lr = train_model_DA(num_epochs=1, batch_size=32, mode="Jigsaw")
    _, _, _, _, losses, accuracies, lr = train_model_source_only(num_epochs=1, batch_size=32, mode="Source")
