import time

import cv2
import numpy as np
import random
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import measurements


class DatasetGenerator:

    def __init__(self, bg_list, fg_list, output_size_range_h=(512, 1024), output_size_range_w=(512, 1024),
                 characters_range=(1, 1), load_all=False):
        self.bg_list = bg_list
        self.fg_list = fg_list
        self.output_size_range_h = output_size_range_h
        self.output_size_range_w = output_size_range_w
        self.mask_erode_kernel = np.ones([3, 3], dtype=np.float32)
        self.load_all = load_all
        self.bgs = []
        self.fgs = []
        random.shuffle(self.bg_list)
        self.bg_idx = 0
        characters_idx = []
        characters_total = 0
        while True:
            if characters_total >= len(fg_list):
                break
            num = random.randint(characters_range[0], characters_range[1])
            characters_idx.append([characters_total + x for x in range(0, num) if characters_total + x < len(fg_list)])
            characters_total += num
        self.characters_idx = characters_idx

        if load_all:
            print("loading bgs")
            for bg_path in tqdm(bg_list):
                bg = cv2.cvtColor(cv2.imread(bg_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                self.bgs.append(bg)
            print("loading fgs")
            for fg_path in tqdm(fg_list):
                fg = cv2.cvtColor(cv2.imread(fg_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                assert fg.shape[2] == 4
                self.fgs.append(fg)

    @staticmethod
    def random_corp(img, out_size=None):
        h, w = img.shape[:2]
        if out_size is None:
            min_s = min(h, w)
            out_size = (min_s, min_s)
        top = random.randint(0, h - out_size[0])
        left = random.randint(0, w - out_size[1])
        img = img[top:top + out_size[0], left:left + out_size[1]]
        return img

    @staticmethod
    def process_fg(fg, output_size):
        assert fg.shape[2] == 4
        h, w = fg.shape[:2]
        if output_size[0] < h or output_size[1] < w:
            r = min(output_size[0] / h, output_size[1] / w)
            new_h, new_w = int(h * r), int(w * r)
            fg = cv2.resize(fg, (new_w, new_h))

        # fg random move
        h, w = output_size
        cy, cx = measurements.center_of_mass(fg[:, :, 3])
        dx = w / 2 - cx
        dy = h / 2 - cy
        fg = cv2.warpAffine(fg, np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
                            output_size[::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
        scale = random.uniform(0.5, 1.5)
        dx = random.randint(-w // 3, w // 3)
        dy = random.randint(-h // 3, h // 3)
        angle = random.randint(-90, 90)
        trans_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        trans_mat[0][2] += dx
        trans_mat[1][2] += dy
        fg = cv2.warpAffine(fg, trans_mat, output_size[::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
        return fg

    def __len__(self):
        return len(self.characters_idx)

    def __getitem__(self, idx):
        output_size = [random.randint(self.output_size_range_h[0], self.output_size_range_h[1]),
                       random.randint(self.output_size_range_w[0], self.output_size_range_w[1])]
        fgs = []
        if self.load_all:
            fgs = [self.fgs[x].astype(np.float32) / 255 for x in self.characters_idx[idx]]
            bg = self.bgs[self.bg_idx].astype(np.float32) / 255
        else:
            fgs = [cv2.cvtColor(cv2.imread(self.fg_list[x], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA).astype(
                np.float32) / 255 for x in self.characters_idx[idx]]
            bg = cv2.cvtColor(cv2.imread(self.bg_list[self.bg_idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255
        self.bg_idx += 1
        if self.bg_idx >= len(self.bg_list):
            self.bg_idx = 0

        # resize to output_size

        h, w = bg.shape[:2]
        if output_size[0] > h or output_size[1] > w:
            r = min(h / output_size[0], w / output_size[1])
            corp_size = (int(output_size[0] * r), int(output_size[1] * r))
            bg = self.random_corp(bg, corp_size)
            bg = cv2.resize(bg, output_size[::-1])
        else:
            bg = self.random_corp(bg, output_size)

        # mix fgs and bg
        image = bg
        label = np.zeros([*output_size, 1], dtype=np.float32)
        for fg in fgs:
            fg = self.process_fg(fg, output_size)
            image_i, label_i = fg[:, :, 0:3], fg[:, :, 3:]
            mask = label_i
            if random.randint(0, 1) == 0:
                mask = cv2.erode(mask, self.mask_erode_kernel)[:, :, np.newaxis]
            image = mask * image_i + (1 - mask) * image
            label = np.fmax(label_i, label)
        label = (label > 0.3).astype(np.float32)

        # random quality
        if random.randint(0, 3) > 0:
            if random.randint(0, 3) == 0:
                image = cv2.blur(image, [3, 3])
            image = Image.fromarray((image * 255).astype(np.uint8))
            image_stream = BytesIO()
            image.save(image_stream, "JPEG", quality=random.randrange(50, 100), optimice=True)
            image_stream.seek(0)
            image = np.asarray(Image.open(image_stream), dtype=np.float32) / 255
        return image, label


if __name__ == "__main__":
    pass
