# This code is modified from the repository
# https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
from PIL import Image, ImageOps

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='RGB')            
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask)

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.BILINEAR)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        # assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        # assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.BILINEAR)

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask): # longer side of image is scaled to defined size.
        warning_size = 5

        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            pass
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)

            if oh < warning_size:
                print('warning: resized image height is less than %d'%warning_size)

            img = img.resize((ow, oh), Image.BILINEAR)
        else:
            oh = self.size
            ow = int(self.size * w / h)

            if ow < warning_size:
                print('warning: resized image width is less than %d'%warning_size)

            img = img.resize((ow, oh), Image.BILINEAR)
        
        w, h = mask.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            pass
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)

            if oh < warning_size:
                print('warning: resized template height is less than %d'%warning_size)

            mask = mask.resize((ow, oh), Image.BILINEAR)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            
            if ow < warning_size:
                print('warning: resized template width is less than %d'%warning_size)
            
            mask = mask.resize((ow, oh), Image.BILINEAR)

        return img, mask

class CenterPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask): # longer side of image is scaled to defined size.
        w, h = img.size

        assert self.size[0] >= h
        assert self.size[1] >= w
        w_pad = self.size[1]-w
        w_pad_left = int(w_pad/2)
        w_pad_right = w_pad - w_pad_left
        h_pad = self.size[0]-h
        h_pad_up = int(h_pad/2)
        h_pad_bottom = h_pad - h_pad_up
        padding = (w_pad_left, h_pad_up, w_pad_right, h_pad_bottom)

        img = ImageOps.expand(img, border=padding, fill=0)        

        w, h = mask.size

        assert self.size[0] >= h
        assert self.size[1] >= w
        w_pad = self.size[1]-w
        w_pad_left = int(w_pad/2)
        w_pad_right = w_pad - w_pad_left
        h_pad = self.size[0]-h
        h_pad_up = int(h_pad/2)
        h_pad_bottom = h_pad - h_pad_up
        padding = (w_pad_left, h_pad_up, w_pad_right, h_pad_bottom)

        mask = ImageOps.expand(mask, border=padding, fill=0) 

        return img, mask

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        # assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.BILINEAR)
        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.BILINEAR)

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.BILINEAR)

        return self.crop(*self.scale(img, mask))