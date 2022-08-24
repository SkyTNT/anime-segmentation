# data loader
import glob
import os
import random
import time

import cv2
import torch
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional
from dataset_generator import DatasetGenerator


class RescalePad(object):
    # Rescale the image to max edge equal output_size and pad to square
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[-2:]
        if h != self.output_size or w != self.output_size:
            r = min(self.output_size / h, self.output_size / w)
            new_h, new_w = int(h * r), int(w * r)
            ph = self.output_size - new_h
            pw = self.output_size - new_w
            image = transforms.functional.resize(image, [new_h, new_w])
            image = transforms.functional.pad(image, [pw // 2, ph // 2, pw // 2 + pw % 2, ph // 2 + ph % 2], 0)
            label = transforms.functional.resize(label, [new_h, new_w])
            label = transforms.functional.pad(label, [pw // 2, ph // 2, pw // 2 + pw % 2, ph // 2 + ph % 2], 0)
        return {'image': image, 'label': label}


class Rescale(object):
    # Rescale the image to min edge equal output_size
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[-2:]
        if min(h, w) != self.output_size:
            r = max(self.output_size / h, self.output_size / w)
            new_h, new_w = int(h * r), int(w * r)
            image = transforms.functional.resize(image, [new_h, new_w])
            label = transforms.functional.resize(label, [new_h, new_w])
        return {'image': image, 'label': label}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        new_h, new_w = self.output_size

        h, w = image.shape[-2:]
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[:, top: top + new_h, left: left + new_w]
        label = label[:, top: top + new_h, left: left + new_w]

        return {'image': image, 'label': label}


class RandomColor(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.randint(0, 1) == 0:
            if random.randint(0, 1) == 0:
                image = transforms.functional.rgb_to_grayscale(image, 3)
            image = transforms.functional.adjust_brightness(image, random.choice([0.5, 1.2]))
            image = transforms.functional.adjust_contrast(image, random.choice([0.5, 1.5]))
        return {'image': image, 'label': label}


class GaussianNoise(object):

    def __init__(self, mean=0, sigma=0.1):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.randint(0, 1) == 0:
            noise = torch.normal(self.mean, self.sigma, image.shape)
            image = image + noise
            image = image.clip(0, 1)
        return {'image': image, 'label': label}


class WithTrimap(object):
    # Generate trimap from mask
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        trimap = label[0].clone().numpy()
        h, w = trimap.shape
        s = int((h + w) * 0.025)
        trimap[np.where((grey_dilation(trimap, size=(s, s)) - grey_erosion(trimap, size=(s, s))) > 0.5)] = 0.5
        trimap = trimap[np.newaxis, :, :]
        trimap = torch.from_numpy(trimap)
        return {'image': image, 'label': label, 'trimap': trimap}


class AnimeSegDataset(Dataset):
    def __init__(self, real_img_list, real_mask_list, generator: DatasetGenerator = None,
                 transform=None, transform_generator=None, with_trimap=False):
        self.dataset_generator = generator
        self.real_img_list = real_img_list
        self.real_mask_list = real_mask_list
        self.transform = transform
        self.transform_generator = transform_generator
        self.with_trimap = WithTrimap() if with_trimap else None

    def __len__(self):
        length = len(self.real_img_list)
        if self.dataset_generator is not None:
            length += len(self.dataset_generator)
        return length

    def __getitem__(self, idx):
        if idx < len(self.real_img_list):
            image = cv2.cvtColor(cv2.imread(self.real_img_list[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            label = cv2.imread(self.real_mask_list[idx], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
            image, label = image.astype(np.float32) / 255, label.astype(np.float32) / 255
            label = (label > 0.3).astype(np.float32)
        else:
            image, label = self.dataset_generator[idx - len(self.real_img_list)]
        image, label = torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(label).permute(2, 0, 1)
        sample = {'image': image, 'label': label}
        if self.transform and idx < len(self.real_img_list):
            sample = self.transform(sample)
        if self.transform_generator and idx >= len(self.real_img_list):
            sample = self.transform_generator(sample)
        if self.with_trimap:
            sample = self.with_trimap(sample)
        return sample


def create_training_datasets(data_root, fgs_dir, bgs_dir, imgs_dir, masks_dir, fg_ext, bg_ext, img_ext, mask_ext,
                             spilt_rate, image_size, with_trimap=False):
    def add_sep(path):
        if not (path.endswith("/") or path.endswith("\\")):
            return path + os.sep
        else:
            return path

    data_root = add_sep(data_root)
    fgs_dir = add_sep(fgs_dir)
    bgs_dir = add_sep(bgs_dir)
    imgs_dir = add_sep(imgs_dir)
    masks_dir = add_sep(masks_dir)

    train_img_list = glob.glob(data_root + imgs_dir + '*' + img_ext)
    train_mask_list = []
    for img_path in train_img_list:
        train_mask_list.append(data_root + masks_dir + img_path.split(os.sep)[-1].replace(img_ext, mask_ext))
    train_fg_list = glob.glob(data_root + fgs_dir + '*' + fg_ext)
    train_bg_list = glob.glob(data_root + bgs_dir + '*' + bg_ext)
    train_fg_list, val_fg_list = train_fg_list[:int(len(train_fg_list) * spilt_rate)], \
                                 train_fg_list[int(len(train_fg_list) * spilt_rate):]
    train_bg_list, val_bg_list = train_bg_list[:int(len(train_bg_list) * spilt_rate)], \
                                 train_bg_list[int(len(train_bg_list) * spilt_rate):]
    train_img_list, val_img_list = train_img_list[:int(len(train_img_list) * spilt_rate)], \
                                   train_img_list[int(len(train_img_list) * spilt_rate):]
    train_mask_list, val_mask_list = train_mask_list[:int(len(train_mask_list) * spilt_rate)], \
                                     train_mask_list[int(len(train_mask_list) * spilt_rate):]
    print("---")
    print("train fgs: ", len(train_fg_list))
    print("train bgs: ", len(train_bg_list))
    print("train imgs: ", len(train_img_list))
    print("train masks: ", len(train_mask_list))
    print("val fgs: ", len(val_fg_list))
    print("val bgs: ", len(val_bg_list))
    print("val imgs: ", len(val_img_list))
    print("val masks: ", len(val_mask_list))
    print("---")
    transform = transforms.Compose([RescalePad(image_size + image_size // 4), RandomCrop(image_size),
                                    RandomColor(), GaussianNoise()])
    transform_generator = transforms.Compose([RandomColor(), GaussianNoise()])
    train_generator = DatasetGenerator(train_bg_list, train_fg_list, (image_size, image_size), (image_size, image_size))
    train_dataset = AnimeSegDataset(train_img_list, train_mask_list, train_generator,
                                    transform=transform, transform_generator=transform_generator,
                                    with_trimap=with_trimap)
    val_generator = DatasetGenerator(val_bg_list, val_fg_list, (image_size, image_size), (image_size, image_size))
    val_dataset = AnimeSegDataset(val_img_list, val_mask_list, val_generator,
                                  transform=transform, transform_generator=transform_generator,
                                  with_trimap=with_trimap)

    return train_dataset, val_dataset
