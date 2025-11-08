import os
import numpy as np
import os.path as osp

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class MiniImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, index_path=None, index=None, base_sess=None, min_scale=None):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.IMAGE_PATH = os.path.join(root, 'mini_imagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'mini_imagenet/split')
        self.min_scale = min_scale

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        train_transform, test_transform = self.get_mini_transforms()
        # use multi-crop augmentation for training
        if transform is not None:
            train_transform = transform
        self.transform = train_transform if train else test_transform
        if train:
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])

        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets

    def get_mini_transforms(self):
        img_size = 84
        _img_resize_size = 96
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(self.min_scale, 1.), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        test_transform = transforms.Compose([
            transforms.Resize([_img_resize_size, _img_resize_size]),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_transform = TwoCropsTransform(train_transform)

        return train_transform, test_transform

class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]