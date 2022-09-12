# data loader
import glob
import os
import random
import time

import cv2
import torch
import numpy as np
import ctypes
import multiprocessing as mp
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


class RandomRot(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        angle = random.uniform(-90, 90)
        image = transforms.functional.rotate(torch.cat([image, label], dim=0), angle,
                                             interpolation=transforms.InterpolationMode.BILINEAR,
                                             fill=[0, 0, 0, 0])
        image, label = image[:-1], image[-1:]
        return {'image': image, 'label': label}


class RandomColor(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        low_definition = False
        if random.randint(0, 1) == 0:
            b = random.choice([random.uniform(0.4, 0.5), random.uniform(1, 1.2)])
            c = random.choice([random.uniform(0.4, 0.5), random.uniform(1, 1.5)])
            image = transforms.functional.adjust_brightness(image, b)
            image = transforms.functional.adjust_contrast(image, c)
            low_definition = b <= 0.5 and c <= 0.5
        return {'image': image, 'label': label, "low_definition": low_definition}


class GaussianNoise(object):

    def __init__(self, mean=0, sigma=0.05):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, sample):
        image, label, low_definition = sample['image'], sample['label'], sample["low_definition"]
        if random.randint(0, 1) == 0 and not low_definition:
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
        trimap[grey_dilation(trimap, size=(s, s)) - grey_erosion(trimap, size=(s, s)) > 0.5] = 0.5
        trimap = trimap[np.newaxis, :, :]
        trimap = torch.from_numpy(trimap)
        return {'image': image, 'label': label, 'trimap': trimap}


class AnimeSegDataset(Dataset):
    def __init__(self, real_img_list, real_mask_list, generator: DatasetGenerator = None,
                 transform=None, transform_generator=None, with_trimap=False, cache_ratio=0.0, cache_life=3):
        self.dataset_generator = generator
        self.real_img_list = real_img_list
        self.real_mask_list = real_mask_list
        self.transform = transform
        self.transform_generator = transform_generator
        self.with_trimap = WithTrimap() if with_trimap else None

        if generator is not None:
            assert generator.output_size_range_w[0] == generator.output_size_range_w[1] \
                   and generator.output_size_range_h[0] == generator.output_size_range_h[1]

        self.use_cache = False
        if cache_ratio > 0:
            assert cache_ratio <= 1
            self.cache_life = cache_life

            h, w = self.get_img_size()
            c = 5 if with_trimap else 4
            n = int(self.__len__() * cache_ratio)
            shared_cache_base = mp.Array(ctypes.c_uint8, n * c * h * w)
            shared_cache = np.ctypeslib.as_array(shared_cache_base.get_obj())
            shared_cache = shared_cache.reshape(n, c, h, w)
            self.shared_cache = torch.from_numpy(shared_cache)
            cacheable_samples = random.Random(1).sample(list(range(0, self.__len__())), n)
            self.cache_idx = [-1] * self.__len__()
            for i, x in enumerate(cacheable_samples):
                self.cache_idx[x] = i
            self.cache_use_count = mp.Array(ctypes.c_int, self.__len__())
            self.use_cache = True

    def __len__(self):
        length = len(self.real_img_list)
        if self.dataset_generator is not None:
            length += len(self.dataset_generator)
        return length

    def get_img_size(self):
        h, w = self.__getitem__(0)['image'].shape[1:]
        return h, w

    def __getitem__(self, idx):
        if self.use_cache and self.cache_idx[idx] != -1 \
                and self.cache_use_count[idx] != 0 and self.cache_use_count[idx] < self.cache_life:
            i = self.cache_idx[idx]
            cache = self.shared_cache[i].float() / 255
            if self.with_trimap:
                sample = {'image': cache[0:3], 'label': cache[3:4], 'trimap': cache[4:5]}
            else:
                sample = {'image': cache[0:3], 'label': cache[3:4]}
            self.cache_use_count[idx] += 1
            return sample

        if idx < len(self.real_img_list):
            image = cv2.cvtColor(cv2.imread(self.real_img_list[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            label = cv2.imread(self.real_mask_list[idx], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
            image, label = image.astype(np.float32) / 255, label.astype(np.float32) / 255
            label = (label > 0.3).astype(np.float32)
            image = image[10:-10, 10:-10]
            label = label[10:-10, 10:-10]  # in this dataset, there is a problem in the edge of some label
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

        if self.use_cache and self.cache_idx[idx] != -1:
            i = self.cache_idx[idx]
            image, label = sample['image'], sample['label']
            if self.with_trimap:
                trimap = sample['trimap']
                cache = torch.cat([image, label, trimap], dim=0)
            else:
                cache = torch.cat([image, label], dim=0)
            cache = (cache * 255).to(torch.uint8)
            self.shared_cache[i] = cache
            self.cache_use_count[idx] = 1
        return sample


def create_training_datasets(data_root, fgs_dir, bgs_dir, imgs_dir, masks_dir, fg_ext, bg_ext, img_ext, mask_ext,
                             spilt_rate, image_size, with_trimap=False, cache_ratio=0.0, cache_update_epoch=3):
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
    random.Random(1).shuffle(train_fg_list)
    random.Random(1).shuffle(train_bg_list)
    random.Random(1).shuffle(train_img_list)
    random.Random(1).shuffle(train_mask_list)
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
    transform = transforms.Compose([RescalePad(image_size + image_size // 4), RandomRot(), RandomCrop(image_size),
                                    RandomColor(), GaussianNoise()])
    transform_generator = transforms.Compose([RandomColor(), GaussianNoise()])
    train_generator = DatasetGenerator(train_bg_list, train_fg_list, (image_size, image_size), (image_size, image_size))
    train_dataset = AnimeSegDataset(train_img_list, train_mask_list, train_generator,
                                    transform=transform, transform_generator=transform_generator,
                                    with_trimap=with_trimap, cache_ratio=cache_ratio, cache_life=cache_update_epoch)
    val_generator = DatasetGenerator(val_bg_list, val_fg_list, (image_size, image_size), (image_size, image_size))
    val_dataset = AnimeSegDataset(val_img_list, val_mask_list, val_generator,
                                  transform=transform, transform_generator=transform_generator,
                                  with_trimap=with_trimap, cache_ratio=0)
    return train_dataset, val_dataset
