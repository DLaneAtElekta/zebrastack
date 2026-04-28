import logging
from typing import List
import numpy as np
from torchvision import transforms
import cv2


def sigmoid(img: np.ndarray, std_weight: float):
    img_avg = np.average(img)
    img_std = np.std(img)

    # sigmoid transform
    img = img - img_avg
    img = img / (std_weight * img_std)
    img = 1.0 / (1.0 + np.exp(-img))
    # logging.debug(f"image sigmoid: min={np.min(img)}, max={np.max(img)}")

    return img


def float_to_uint8(img: np.ndarray):
    assert np.min(img) >= 0.0
    assert np.max(img) <= 1.0
    img_uint8 = img * 255.0
    img_uint8 = np.clip(img_uint8, 0.0, 255.0)
    img_uint8 = img_uint8.astype(np.uint8)
    return img_uint8


def uint8_to_float(img):
    min, max = np.min(img), np.max(img)
    # logging.debug(f"uint8_to_float: image min/max = {min}, {max}")
    img = (img - min) / (max - min)
    img = img.astype(np.float32)
    return img


class ClaheTransform(transforms.Lambda):
    def __init__(self, clip_limit: int, grid_sizes: List[int]):
        super().__init__(self.__call__)
        self.clip_limit = clip_limit
        self.std_weight = 0.66
        self.clahe_filters = [
            cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            for grid_size in grid_sizes
        ]

    def __call__(self, img):
        img = sigmoid(img)

        # convert to uint8
        img_uint8 = float_to_uint8(img)

        # apply clahe stack
        filtered_imgs = []
        for clahe in self.clahe_filters:
            filtered_img = clahe.apply(img_uint8)
            filtered_img = uint8_to_float(filtered_img)
            filtered_imgs.append(filtered_img)

        return np.stack([img] + filtered_imgs, axis=-1)
