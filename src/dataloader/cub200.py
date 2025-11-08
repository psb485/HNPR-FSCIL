import os
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CUB200(Dataset):
    def __init__(self, root, train=True, index_path=None, index=None, base_sess=None, transform=None, min_scale=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self._pre_operate(self.root)
        self.min_scale = min_scale
 
        train_transform, test_transform = self.get_cub_transforms()
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

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        data_tmp = []
        targets_tmp = []
        index = open(index_path).read().splitlines()
        for i in index:
            img_path = os.path.join(self.root, i)
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

    def get_cub_transforms(self):
        img_size = 224
        _img_resize_size = 256
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(self.min_scale, 1.), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.75),  
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.Resize(_img_resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_transform = TwoCropsTransform(test_transform, train_transform)

        return train_transform, test_transform

class TwoCropsTransform:
    def __init__(self, original_transform, augment_transform):
        self.original_transform = original_transform
        self.augment_transform = augment_transform

    def __call__(self, x):
        q = self.original_transform(x)
        k = self.augment_transform(x)
        return [q, k]