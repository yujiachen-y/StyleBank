import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from image_folder_functions import is_photo

data_dir = 'datasets'


##################
# (1) ImageLoader
##################
class ImageLoader(object):
    def __init__(self, content_dir, style_dir):
        assert os.path.isdir(content_dir) and os.path.isdir(style_dir)

        self.content_dir, self.style_dir = content_dir, style_dir
        length1, length2 = 0, 0
        content_list, style_list = [], []

        files = os.listdir(content_dir)
        for file in files:
            if is_photo(file):
                length1 += 1
                content_list.append(file)
        files = os.listdir(style_dir)
        for file in files:
            if is_photo(file):
                length2 += 1
                style_list.append(file)

        assert length1 == length2
        self.length = length1
        self.content_list, self.style_list = content_list, style_list

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        assert 0 <= item < self.length
        content_image_dir = os.path.join(self.content_dir,
                                         self.content_list[item])
        style_image_dir = os.path.join(self.style_dir, self.style_list[item])
        content_image = Image.open(content_image_dir).convert('RGB')
        style_image = Image.open(style_image_dir).convert('RGB')
        return content_image, style_image


####################################
# (2) combine DataLoaders to Dataset
####################################
class ContentStyleDataset(Dataset):
    def __init__(self, dataloaders, transform_list):
        self.length = 0
        for dataloader in dataloaders:
            self.length += len(dataloader)
        self.dataloaders = dataloaders
        self.transform_list = transform_list
        self.horizontal_flip = transforms.RandomHorizontalFlip(1)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        assert 0 <= item < self.length
        style_id = 1
        for dataloader in self.dataloaders:
            if item >= len(dataloader):
                item -= len(dataloader)
                style_id += 1
            else:
                break
        content_image, style_image =self.dataloaders[style_id-1][item]

        if np.random.rand() < 0.5:
            content_image = self.horizontal_flip(content_image)
            style_image = self.horizontal_flip(style_image)
        trans_id = np.random.randint(len(self.transform_list))
        content_image = self.transform_list[trans_id](content_image)
        style_image = self.transform_list[trans_id](style_image)
        return style_id, content_image, style_image


# class CocoDataset(Dataset):
#     def __init__(self, transform):
#         self.transform = transform
#         self.img_list = []
#         img_dir = os.path.join(data_dir, 'Top_1000_pictures_in_COCO_2017val')
#         dir_list = os.listdir(img_dir)
#
#         for file in dir_list:
#             if is_photo(file):
#                 self.img_list.append(os.path.join(img_dir, file))
#
#     def __len__(self):
#         return len(self.img_list)
#
#     def __getitem__(self, item):
#         img = Image.open(self.img_list[item]).convert('RGB')
#         img = self.transform(img)
#         return img
