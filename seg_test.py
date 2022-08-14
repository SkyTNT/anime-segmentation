import os

import cv2
import torch

import numpy as np
import glob

from skimage import io
from torch.cuda import amp
from tqdm import tqdm

from data_loader import SalObjDataset
from dataset_generator import DatasetGenerator
from model import U2NET_full2, U2NET_lite2, ISNetDIS, U2NET


def get_mask(input_img, use_amp=True, s=640):
    if input_img.shape[2] == 4:
        input_img = input_img[:, :, 0:3]
    h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
    ph, pw = s - h, s - w
    input_img = cv2.resize(input_img, (w, h)) / 255
    tmpImg = np.zeros([s, s, 3], dtype=np.float32)
    tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = input_img
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                if isinstance(net, U2NET):
                    d1, _, _, _, _, _, _ = net(tmpImg)
                elif isinstance(net, ISNetDIS):
                    ds, _ = net(tmpImg)
                    d1 = ds[0]
                d1 = torch.sigmoid(d1)
            d1 = d1.to(dtype=torch.float32)
        else:
            if isinstance(net, U2NET):
                d1, _, _, _, _, _, _ = net(tmpImg)
            elif isinstance(net, ISNetDIS):
                ds, _ = net(tmpImg)
                d1 = ds[0]
            d1 = torch.sigmoid(d1)
        pred = d1[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(cv2.cvtColor(pred.cpu().numpy().transpose((1, 2, 0)), cv2.COLOR_GRAY2RGB),
                          (w0, h0))
        return pred


if __name__ == "__main__":
    net = ISNetDIS()
    device = torch.device('cuda:0')
    net.load_state_dict(torch.load('saved_models/isnet_best.pt', device))
    net.to(device)
    net.eval()
    gen_data = True
    size = 1024

    if gen_data:
        data_dir = '../../dataset/anime-seg/'
        tra_fg_dir = 'fg/'
        tra_bg_dir = 'bg/'
        fg_ext = '.png'
        bg_ext = '.*'
        tra_fg_list = glob.glob(data_dir + tra_fg_dir + '*' + fg_ext)
        tra_bg_list = glob.glob(data_dir + tra_bg_dir + '*' + bg_ext)
        generator = DatasetGenerator(tra_bg_list, tra_fg_list, output_size_range_h=(size, size),
                                     output_size_range_w=(size, size), characters_range=(1,1))
        salobj_dataset = SalObjDataset([], [], generator)
        for i, data in enumerate(tqdm(salobj_dataset)):
            img = (data['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask = get_mask(img, s=size)
            img = np.concatenate((img, mask * img, mask * 255), axis=1).astype(np.uint8)
            io.imsave(f'out/{i:06d}.jpg', img)
    else:
        for i, path in enumerate(tqdm(glob.glob("../../dataset/anime-seg/test2/*.*"))):
            img = io.imread(path)
            if img.shape[2] == 4:
                img = img[:, :, 0:3]
            mask = get_mask(img, s=size)
            img = np.concatenate((img, mask * img, mask * 255), axis=1).astype(np.uint8)
            io.imsave(f'out/{i:06d}.jpg', img)
