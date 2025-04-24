"""
get data loaders
"""
from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import json

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets/imagenet'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data/imagenet'
    else:
        data_folder = '/root/RepVGG/wsl/data/mini-imagenet'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class MyDataSet_mini(Dataset):
    """自定义数据集"""

    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 transform=None):
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)
        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)

        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class ImageFolderInstance(MyDataSet_mini):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target, index


class ImageFolderSample(MyDataSet_mini):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.k = k
        self.is_sample = is_sample

        print('stage1 finished!')

        if self.is_sample:
            num_classes = 100
            num_samples = len(self.img_paths)
            label = self.img_label
            

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        img = Image.open(self.img_paths[index])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[index]))
        target = self.img_label[index]

        if self.transform is not None:
            img = self.transform(img)


     

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_test_loader(dataset='miniImagenet', batch_size=128, num_workers=8):
    """get the test data loader"""

    if dataset == 'miniImagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_set = MyDataSet_mini(root_dir=data_folder,
                            csv_name="new_val.csv",
                            json_path="/root/distill/data/mini-imagenet/classes_name.json",
                            transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return test_loader


def get_dataloader_sample(dataset='imagenet', batch_size=128, num_workers=8, is_sample=False, k=4096):
    """Data Loader for ImageNet"""

    if dataset == 'miniImage':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    # add data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_set = MyDataSet_mini(root_dir=data_folder,
                            csv_name="new_train.csv",
                            json_path="/root/distill/data/mini-imagenet/classes_name.json",
                            transform=train_transform)

    test_set = MyDataSet_mini(root_dir=data_folder,
                            csv_name="new_val.csv",
                            json_path="/root/distill/data/mini-imagenet/classes_name.json",
                            transform=test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers//2,
                             pin_memory=True)

    print('num_samples', len(train_set.img_paths))
    print('num_class', len(train_set.labels))

    return train_loader, test_loader, len(train_set), len(train_set.labels)


def get_miniImagenet_dataloader(dataset='miniImage', batch_size=128, num_workers=16, is_instance=False):
    """
    Data Loader for imagenet
    """
    if dataset == 'miniImage':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])



    if is_instance:
        train_set = ImageFolderInstance(root_dir=data_folder,
                              csv_name="new_train.csv",
                              json_path="/root/distill/data/mini-imagenet/classes_name.json",
                              transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = MyDataSet_mini(root_dir=data_folder,
                              csv_name="new_train.csv",
                              json_path="/root/distill/data/mini-imagenet/classes_name.json",
                              transform=train_transform)

    test_set = MyDataSet_mini(root_dir=data_folder,
                            csv_name="new_val.csv",
                            json_path="/root/distill/data/mini-imagenet/classes_name.json",
                            transform=test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers//2,
                             pin_memory=True)

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader
