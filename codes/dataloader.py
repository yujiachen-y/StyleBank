import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from image_folder_functions import is_photo

data_dir = 'datasets'


class StyleBankDataset(Dataset):
    def __init__(self, transform, style_info, style_list):
        self.transform = transform
        self.style_info = style_info
        self.style_list = style_list
        self.length = 0
        for style in style_list:
            self.length += style_info[style]['length']


    def __len__(self):
        return self.length


    def __getitem__(self, item):
        style_id, original_image, stylize_image = 0, None, None
        style_info = self.style_info
        for style in self.style_list:
            if item >= style_info[style]['length']:
                item -= style_info[style]['length']
                style_id = style_id + 1
            else:
                info = style_info[style]
                original_image_dir = os.path.join(data_dir, style, 'photos', info['photos'][item])
                stylize_image_dir = os.path.join(data_dir, style, 'sketches', info['sketches'][item])
                original_image = Image.open(original_image_dir).convert('RGB')
                stylize_image = Image.open(stylize_image_dir).convert('RGB')
                break

        original_image = self.transform(original_image)
        stylize_image = self.transform(stylize_image)

        return (style_id, original_image, stylize_image)


class CocoDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.img_list = []
        img_dir = os.path.join(data_dir, 'Top_1000_pictures_in_COCO_2017val')
        dir_list = os.listdir(img_dir)

        for file in dir_list:
            if is_photo(file):
                self.img_list.append(os.path.join(img_dir, file))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img = Image.open(self.img_list[item]).convert('RGB')
        img = self.transform(img)
        return img
