import math
import time

import cv2
import numpy as np
import random
import ctypes
import multiprocessing as mp
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from scipy.ndimage import measurements


def vector_included_angle(v1, v2):
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    a = a1 - a2
    if a > math.pi:
        a = a - math.pi * 2
    if a < -math.pi:
        a = a + math.pi * 2
    return a


class DatasetGenerator:

    def __init__(self, bg_list, fg_list, output_size_range_h=(512, 1024), output_size_range_w=(512, 1024),
                 characters_range=(0, 3), seed=1, load_all=False):
        self.bg_list = bg_list
        self.fg_list = fg_list
        self.output_size_range_h = output_size_range_h
        self.output_size_range_w = output_size_range_w
        self.load_all = load_all
        self.bgs = []
        self.fgs = []
        characters_idx = []
        characters_total = 0
        self.random = random.Random(seed)
        while True:
            if characters_total >= len(fg_list):
                break
            num = self.random.randint(characters_range[0], characters_range[1])
            characters_idx.append([characters_total + x for x in range(0, num) if characters_total + x < len(fg_list)])
            characters_total += num
        self.characters_idx = characters_idx

        self.texts = [chr(x) for x in range(0x3040, 0x30ff + 1)]
        self.fonts = []

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
        self.bgs_offset = mp.Array(ctypes.c_long, self.__len__())

    def random_corp(self, img, out_size=None):
        h, w = img.shape[:2]
        if out_size is None:
            min_s = min(h, w)
            out_size = (min_s, min_s)
        top = self.random.randint(0, h - out_size[0])
        left = self.random.randint(0, w - out_size[1])
        img = img[top:top + out_size[0], left:left + out_size[1]]
        return img

    def process_fg(self, fg, output_size, scale_range=(0.2, 0.8)):
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
        scale = self.random.uniform(*scale_range)
        dx = self.random.randint(-w // 3, w // 3)
        dy = self.random.randint(-h // 3, h // 3)
        angle = self.random.randint(-90, 90)
        trans_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        trans_mat[0][2] += dx
        trans_mat[1][2] += dy
        fg = cv2.warpAffine(fg, trans_mat, output_size[::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
        return fg

    def simulate_light(self, image, strength=0.2):
        img_size = image.shape[:2]
        a = int(np.linalg.norm(img_size) / 2)
        r = self.random.randint(a * 11 // 10, a * 2)
        b = self.random.uniform(0, math.pi * 2)
        cx = int(img_size[1] // 2 + r * math.cos(b))
        cy = int(img_size[0] // 2 + r * math.sin(b))
        c_v = [img_size[1] // 2 - cx, img_size[0] // 2 - cy]
        rs = [vector_included_angle([-cx, -cy], c_v),
              vector_included_angle([img_size[1] - cx, -cy], c_v),
              vector_included_angle([-cx, img_size[0] - cy], c_v),
              vector_included_angle([img_size[1] - cx, img_size[0] - cy], c_v)]
        ds = [np.linalg.norm([-cx, -cy]),
              np.linalg.norm([img_size[1] - cx, -cy]),
              np.linalg.norm([-cx, img_size[0] - cy]),
              np.linalg.norm([img_size[1] - cx, img_size[0] - cy])]
        r2 = max(ds)
        cr = math.atan2(c_v[1], c_v[0])
        if cr < 0:
            cr = math.pi * 2 + cr
        sr = min(rs) + cr
        er = max(rs) + cr
        n = int(50 * (er - sr) * 2 / math.pi)
        color = (self.random.uniform(1 - strength, 1),
                 self.random.uniform(1 - strength, 1),
                 self.random.uniform(1 - strength, 1))
        if self.random.randint(0, 1) == 0:
            light_mask = np.full([*img_size, 3], (1 + strength, 1 + strength, 1 + strength), dtype=np.float32)
        else:
            light_mask = np.full([*img_size, 3], color, dtype=np.float32)
            color = (1 + strength, 1 + strength, 1 + strength)
        for a in np.linspace(sr, er, num=n):
            x2 = int(cx + r2 * math.cos(a))
            y2 = int(cy + r2 * math.sin(a))
            light_mask = cv2.line(light_mask, [cx, cy], [x2, y2], color, 10)
        return (image * light_mask).clip(0, 1)

    def __len__(self):
        return len(self.characters_idx)

    def __getitem__(self, idx):
        # to traverse backgrounds
        bg_idx = (idx + self.bgs_offset[idx]) % len(self.bg_list)
        self.bgs_offset[idx] += 1

        output_size = [self.random.randint(self.output_size_range_h[0], self.output_size_range_h[1]),
                       self.random.randint(self.output_size_range_w[0], self.output_size_range_w[1])]

        if self.load_all:
            fgs = [self.fgs[x].astype(np.float32) / 255 for x in self.characters_idx[idx]]
            bg = self.bgs[bg_idx].astype(np.float32) / 255
        else:
            fgs = [cv2.cvtColor(cv2.imread(self.fg_list[x], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA).astype(
                np.float32) / 255 for x in self.characters_idx[idx]]
            bg = cv2.cvtColor(cv2.imread(self.bg_list[bg_idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255

        # resize to output_size

        h, w = bg.shape[:2]
        if output_size[0] > h or output_size[1] > w:
            r = min(h / output_size[0], w / output_size[1])
            corp_size = (int(output_size[0] * r), int(output_size[1] * r))
            bg = self.random_corp(bg, corp_size)
            bg = cv2.resize(bg, output_size[::-1])
        else:
            bg = self.random_corp(bg, output_size)

        # aug = self.random.randint(0, 1) == 0
        aug = True

        if aug and self.random.randint(0, 1) == 0:
            # generate sharp background
            d = 50
            counts = []
            ms = max(output_size)
            for i in range(0, d):
                r = self.random.randint(ms * 2 // 10, ms * 6 // 10)
                x = output_size[1] // 2 + r * math.cos(math.radians(i / d * 360))
                y = output_size[0] // 2 + r * math.sin(math.radians(i / d * 360))
                counts.append([x, y])
            counts = [np.array(counts, dtype=np.int)]
            bg_mask = cv2.drawContours(np.zeros([*output_size, 1], dtype=np.float32), counts, 0, (1.0,), cv2.FILLED)
            bg = bg * bg_mask + 1 - bg_mask
            if self.random.randint(0, 1) == 0:
                edge_color = (self.random.uniform(0, 1), self.random.uniform(0, 1), self.random.uniform(0, 1))
                bg = cv2.drawContours(bg, counts, 0, edge_color, self.random.randint(ms // 600, ms // 400))

        if aug and self.random.randint(0, 1) == 0:
            # random color blocks on background
            temp_img = np.zeros([*output_size, 4], dtype=np.float32)
            for _ in range(0, 10):
                if self.random.randint(0, 1) == 0:
                    w = self.random.randint(output_size[1] // 20, output_size[1] // 3)
                    h = self.random.randint(output_size[0] // 20, output_size[0] // 3)
                    x = self.random.randint(0, output_size[1] - w)
                    y = self.random.randint(0, output_size[0] - h)
                    color = (self.random.uniform(0, 1), self.random.uniform(0, 1), self.random.uniform(0, 1), 1)
                    temp_img = cv2.rectangle(temp_img, [x, y], [x + w, y + h], color, cv2.FILLED)
                    if self.random.randint(0, 1) == 0:
                        color = (color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, color[2])
                        s = output_size[0] + output_size[0]
                        temp_img = cv2.rectangle(temp_img, [x, y], [x + w, y + h], color,
                                                 self.random.randint(s // 500, s // 400))
                else:
                    r = self.random.randint((output_size[0] + output_size[0]) // 40,
                                            (output_size[0] + output_size[0]) // 8)
                    x = self.random.randint(r, output_size[1] - r)
                    y = self.random.randint(r, output_size[0] - r)
                    color = (self.random.uniform(0, 1), self.random.uniform(0, 1),
                             self.random.uniform(0, 1), self.random.uniform(0.3, 0.5))
                    temp_img = cv2.circle(temp_img, [x, y], r, color, cv2.FILLED)
                    if self.random.randint(0, 1) == 0:
                        color = (color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 1)
                        s = output_size[0] + output_size[0]
                        temp_img = cv2.circle(temp_img, [x, y], r, color, self.random.randint(s // 500, s // 400))
            angle = self.random.randint(-90, 90)
            trans_mat = cv2.getRotationMatrix2D((output_size[1] // 2, output_size[0] // 2), angle, 1)
            temp_img = cv2.warpAffine(temp_img, trans_mat, output_size[::-1], flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
            temp_img, mask = temp_img[:, :, 0:3], temp_img[:, :, 3:]
            bg = mask * temp_img + (1 - mask) * bg

        # mix fgs and bg
        image = bg
        label = np.zeros([*output_size, 1], dtype=np.float32)
        for fg in fgs:
            fg = self.process_fg(fg, output_size, (0.8, 1.2) if len(fgs) == 1 else (0.2, 0.8))
            image_i, label_i = fg[:, :, 0:3], fg[:, :, 3:]
            mask = label_i
            if self.random.randint(0, 1) == 0:
                mask = cv2.blur(mask, [5, 5])[:, :, np.newaxis]
            image_i = image_i * label_i + 1 - label_i
            image = mask * image_i + (1 - mask) * image
            label = np.fmax(label_i, label)
        label = (label > 0.5).astype(np.float32)

        if aug and self.random.randint(0, 1) == 0:
            image = cv2.medianBlur(image, 3)

        if aug and self.random.randint(0, 1) == 0:
            # random color blocks
            temp_img = np.zeros([*output_size, 4], dtype=np.float32)
            for _ in range(0, 10):
                if self.random.randint(0, 1) == 0:
                    w = self.random.randint(output_size[1] // 20, output_size[1] // 3)
                    h = self.random.randint(output_size[0] // 20, output_size[0] // 3)
                    x = self.random.randint(0, output_size[1] - w)
                    y = self.random.randint(0, output_size[0] - h)
                    color = (self.random.uniform(0, 1), self.random.uniform(0, 1),
                             self.random.uniform(0, 1), self.random.uniform(0.3, 0.5))
                    temp_img = cv2.rectangle(temp_img, [x, y], [x + w, y + h], color, cv2.FILLED)
                else:
                    r = self.random.randint((output_size[0] + output_size[0]) // 40,
                                            (output_size[0] + output_size[0]) // 8)
                    x = self.random.randint(r, output_size[1] - r)
                    y = self.random.randint(r, output_size[0] - r)
                    color = (self.random.uniform(0, 1), self.random.uniform(0, 1),
                             self.random.uniform(0, 1), self.random.uniform(0.3, 0.5))
                    temp_img = cv2.circle(temp_img, [x, y], r, color, cv2.FILLED)
            angle = self.random.randint(-90, 90)
            trans_mat = cv2.getRotationMatrix2D((output_size[1] // 2, output_size[0] // 2), angle, 1)
            temp_img = cv2.warpAffine(temp_img, trans_mat, output_size[::-1], flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
            temp_img, mask = temp_img[:, :, 0:3], temp_img[:, :, 3:]
            image = mask * temp_img + (1 - mask) * image

        if aug and self.random.randint(0, 1) == 0:
            # random texts
            image = Image.fromarray((image * 255).astype(np.uint8))
            draw = ImageDraw.Draw(image)
            for _ in range(0, self.random.randint(1, 10)):
                if len(self.fonts) == 0:
                    self.fonts = [ImageFont.truetype("font.otf", x, encoding="utf-8") for x in range(10, 30, 2)]
                font = self.random.choice(self.fonts)
                s = font.size
                text = "".join([self.random.choice(self.texts) for _ in range(0, 10)])
                x = self.random.randint(0, output_size[1] - s * len(text))
                y = self.random.randint(0, output_size[0] - s)
                if self.random.randint(0, 1) == 0:
                    color = (255, 255, 255)
                else:
                    color = (0, 0, 0)
                draw.text((x, y), text, color, font=font)
            image = np.asarray(image).astype(np.float32) / 255

        if aug and self.random.randint(0, 1) == 0:
            image = self.simulate_light(image)

        if aug and self.random.randint(0, 1) == 0:
            # random border
            mask = np.zeros([*output_size, 1], dtype=np.float32)
            if self.random.randint(0, 1) == 0:
                p1 = [self.random.randint(0, output_size[1] // 10), self.random.randint(0, output_size[0] // 10)]
                p2 = [output_size[1] - self.random.randint(0, output_size[1] // 10),
                      output_size[0] - self.random.randint(0, output_size[0] // 10)]
                mask = cv2.rectangle(mask, p1, p2, (1.0,), cv2.FILLED)
                image = image * mask + 1 - mask
                label = label * mask
                if self.random.randint(0, 1) == 0:
                    s = max(output_size)
                    image = cv2.rectangle(image, p1, p2, (0, 0, 0), self.random.randint(s // 600, s // 400))

            else:
                p = [output_size[1] // 2 + self.random.randint(0, output_size[1] // 50),
                     output_size[0] // 2 + self.random.randint(0, output_size[0] // 50)]
                r = min(output_size) // 2
                r = self.random.randint(r * 9 // 10, r)
                mask = cv2.circle(mask, p, r, (1.0,), cv2.FILLED)
                image = image * mask + 1 - mask
                label = label * mask
                if self.random.randint(0, 1) == 0:
                    s = max(output_size)
                    image = cv2.circle(image, p, r, (0, 0, 0), self.random.randint(s // 600, s // 400))

        # random quality
        if aug and self.random.randint(0, 1) == 0:
            image = cv2.blur(image, [3, 3])
        if aug and self.random.randint(0, 1) == 0:
            image = Image.fromarray((image * 255).astype(np.uint8))
            image_stream = BytesIO()
            image.save(image_stream, "JPEG", quality=self.random.randrange(50, 100), optimice=True)
            image_stream.seek(0)
            image = np.asarray(Image.open(image_stream), dtype=np.float32) / 255
        return image, label


if __name__ == "__main__":
    pass
