from __future__ import print_function, division
from torch.utils.data import Dataset
import torch
import os
import random
from PIL import Image
from itertools import permutations

perms = list(permutations(range(4)))


class CustomImageDataset(Dataset):
    def __init__(self, paths_c, paths_d, img_dir, DoRot=False, labels="",
                 transform=None, ROD=False, Jigsaw=False):
        self.paths_c = paths_c
        self.paths_d = paths_d
        self.len = len(self.paths_c)
        if labels !="" or []:
            self.img_labels = labels
        else:
            self.img_labels = [0] * self.len
        self.img_dir = img_dir
        self.transform = transform
        self.DoRot = DoRot
        self.Jigsaw = Jigsaw
        self.ROD = ROD
        self.z = [0] * self.len
        self.rot_c = [0] * self.len
        self.rot_d = [0] * self.len

        if Jigsaw:
            for i in range(self.len):
                self.z[i] = random.choice(list(range(4*3*2*1)))
        elif DoRot:
            for i in range(self.len):
                self.z[i] = random.choice([0, 1, 2, 3])
                self.rot_c[i] = random.choice([0, 1, 2, 3])
                self.rot_d[i] = self.rot_c[i] - self.z[i]
                if self.rot_d[i] < 0:
                    self.rot_d[i] += 4
        else:
            for i in range(self.len):
                self.z[i] = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.ROD:
            rgb_path = os.path.join(self.img_dir, "rgb-washington/", self.paths_c[idx])
            depth_path = os.path.join(self.img_dir, "surfnorm-washington/", self.paths_d[idx])
        else:
            rgb_path = os.path.join(self.img_dir, self.paths_c[idx])
            depth_path = os.path.join(self.img_dir, self.paths_d[idx])
        image_c = Image.open(rgb_path).convert('RGB')
        image_d = Image.open(depth_path).convert('RGB')
        if self.DoRot:
            image_c_rot = image_c.rotate(self.rot_c[idx] * 90)
            image_d_rot = image_d.rotate(self.rot_d[idx] * 90)
        elif self.Jigsaw:
            img1 = image_c.resize((224, 224))
            img2 = image_d.resize((224, 224))
            blockmap1 = [(0, 0, 112, 112), (0, 112, 112, 224), (112, 0, 224, 112), (112, 112, 224, 224)]
            blockmap2 = [(0, 0, 112, 112), (0, 112, 112, 224), (112, 0, 224, 112), (112, 112, 224, 224)]
            shuffle = list(blockmap1)
            shuffle = list(map(shuffle.__getitem__, perms[self.z[idx]]))
            result1 = Image.new(img1.mode, (224, 224))
            result2 = Image.new(img2.mode, (224, 224))
            for box1, box2, sbox in zip(blockmap1, blockmap2, shuffle):
                c = img1.crop(sbox)
                d = img2.crop(sbox)
                result1.paste(c, box1)
                result2.paste(d, box2)
            image_c_rot = result1
            image_d_rot = result2
        else:
            image_c_rot = Image.new('RGB', (224, 224))
            image_d_rot = Image.new('RGB', (224, 224))
        #if self.img_labels != "" and self.img_labels != []:
        #    label = self.img_labels[idx]
        #else:
        #    label = 0
        if self.transform:
            image_c = self.transform(image_c)
            image_d = self.transform(image_d)
            image_c_rot = self.transform(image_c_rot)
            image_d_rot = self.transform(image_d_rot)
        x = torch.cat((image_c, image_d), 0)
        x_t = torch.cat((image_c_rot, image_d_rot), 0)
        return x, self.img_labels[idx], x_t, self.z[idx]
