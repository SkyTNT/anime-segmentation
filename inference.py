import os

import argparse
import cv2
import torch
import numpy as np
import glob
from torch.cuda import amp
from tqdm import tqdm

from train import AnimeSegmentation


def get_mask(input_img, use_amp=True, s=640, itr=1):
    h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
    ph, pw = s - h, s - w
    tmpImg = np.zeros([s, s, 3], dtype=np.float32)
    tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(cv2.cvtColor(pred.cpu().numpy().transpose((1, 2, 0)), cv2.COLOR_GRAY2RGB),
                          (w0, h0))
    if itr > 1:
        return get_mask(input_img * pred, use_amp, s, itr - 1)
    else:
        return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--net', type=str, default='isnet_is',
                        choices=["isnet_is", "isnet", "u2net", "u2netl", "modnet"],
                        help='net name')
    parser.add_argument('--ckpt', type=str, default='saved_models/isnet_best.ckpt',
                        help='model checkpoint path')
    parser.add_argument('--data', type=str, default='../../dataset/anime-seg/test2',
                        help='output dir')
    parser.add_argument('--out', type=str, default='out',
                        help='output dir')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='input image size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cpu or cuda:0')
    opt = parser.parse_args()
    print(opt)

    model = AnimeSegmentation.load_from_checkpoint(opt.ckpt, net_name=opt.net, strict=False)
    model.eval()
    device = torch.device(opt.device)
    model.to(device)

    if not os.path.exists(opt.out):
        os.mkdir(opt.out)

    for i, path in enumerate(tqdm(glob.glob(f"{opt.data}/*.*"))):
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        mask = get_mask(img, s=opt.img_size)
        img = np.concatenate((img, mask * img, mask * 255), axis=1).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'out/{i:06d}.jpg', img)
