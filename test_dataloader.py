import glob
import os
import time

import cv2
from torch.utils.data import DataLoader
from data_loader import create_training_datasets

import numpy as np

if __name__ == '__main__':
    data_dir = '../../dataset/anime-seg/'
    tra_fg_dir = 'fg/'
    tra_bg_dir = 'bg/'
    tra_img_dir = 'imgs/'
    tra_mask_dir = 'masks/'
    fg_ext = '.png'
    bg_ext = '.*'
    img_ext = '.jpg'
    mask_ext = '.jpg'

    train_dataset, val_dataset = create_training_datasets(data_dir, tra_fg_dir, tra_bg_dir, tra_img_dir, tra_mask_dir,
                                                          fg_ext, bg_ext, img_ext, mask_ext, 0.95, 640, True)

    salobj_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, persistent_workers=True)
    for data in salobj_dataloader:
        cv2.imshow("a", np.concatenate([data['image'][0].permute(1, 2, 0).numpy()[:, :, ::-1],
                                        cv2.cvtColor(data['label'][0].permute(1, 2, 0).numpy(), cv2.COLOR_GRAY2RGB),
                                        cv2.cvtColor(data['trimap'][0].permute(1, 2, 0).numpy(), cv2.COLOR_GRAY2RGB)],
                                       axis=1))
        cv2.waitKey(1000)
