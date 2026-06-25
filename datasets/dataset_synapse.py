import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def random_rot_flip(image, label, binary_label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    binary_label = np.rot90(binary_label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    binary_label = np.flip(binary_label, axis=axis).copy()
    return image, label, binary_label


def random_rotate(image, label, binary_label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    binary_label = ndimage.rotate(binary_label, angle, order=0, reshape=False)
    return image, label, binary_label


def random_crop(image, label, binary_label):
    min_ratio = 0.2
    max_ratio = 0.8

    w, h = image.shape

    ratio = random.random()

    scale = min_ratio + ratio * (max_ratio - min_ratio)

    new_h = int(h * scale)
    new_w = int(w * scale)

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    image = image[x:x+new_w,y:y+new_h]
    label = label[x:x+new_w,y:y+new_h]
    binary_label = binary_label[x:x+new_w,y:y+new_h]

    return image, label, binary_label


def random_scale(image, label, binary_label):
    min_ratio = 0.2
    max_ratio = 0.8
    w, h = image.shape
    ratio = random.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    image = image[x:x+new_w, y:y+new_h]
    label = label[x:x+new_w, y:y+new_h]
    binary_label = binary_label[x:x+new_w, y:y+new_h]
    return image, label, binary_label

def random_elastic(image, label, binary_label, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    label = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    binary_label = cv2.warpAffine(binary_label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    # Generate displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    image = map_coordinates(image, indices, order=1, mode='constant').reshape(shape)
    label = map_coordinates(label, indices, order=1, mode='constant').reshape(shape)
    binary_label = map_coordinates(binary_label, indices, order=1, mode='constant').reshape(shape)
    return image, label, binary_label


def random_gaussian(image, var=0.1):
    noise = np.random.normal(0, var, image.shape)
    image = image + noise
    return image

def random_gaussian_filter(im, K_size=3, sigma=1.3):
    im = im*255
    img = np.asarray(np.uint8(im))
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
 
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
 
    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma) 
    K /= K.sum()
    tmp = out.copy()
 
    # filtering
    for y in range(H):
       for x in range(W):
            for c in range(C): 
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    out = out.astype(np.float32) / 255
    out = np.squeeze(out)
    return out


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label, binary_label = sample['image'], sample['label'], sample['binary_label']

        # 应用数据增强并处理binary_label
        if random.random() > 0.5:
            image, label, binary_label = random_rot_flip(image, label, binary_label)
        elif random.random() > 0.55:
            image, label, binary_label = random_rotate(image, label, binary_label)
        elif random.random() > 0.25:
            image, label, binary_label = random_scale(image, label, binary_label)
        elif random.random() > 0.15:
            image, label, binary_label = random_elastic(image, label, binary_label,
                                                        image.shape[1] * 2, image.shape[1] * 0.08,
                                                        image.shape[1] * 0.08)

        # 调整尺寸
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image_512 = zoom(image, (self.output_size[0]/x, self.output_size[1]/y), order=3)
            label_512 = zoom(label, (self.output_size[0]/x, self.output_size[1]/y), order=0)
            binary_label_512 = zoom(binary_label, (self.output_size[0]/x, self.output_size[1]/y), order=0)
        else:
            image_512 = image
            label_512 = label
            binary_label_512 = binary_label

        #生成低分辨率图像
        image_h, image_w = image.shape
        low_image = zoom(image, (self.low_res[0]/image_h, self.low_res[0]/image_w), order=3)

        # 生成低分辨率标签
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0]/label_h, self.low_res[0]/label_w), order=0)

        binary_label_h, binary_label_w = binary_label.shape
        low_binary_label = zoom(binary_label, (self.low_res[0]/binary_label_h, self.low_res[0]/binary_label_w), order=0)


        # 转换为张量
        image_512 = torch.from_numpy(image_512.astype(np.float32)).unsqueeze(0)
        image_512 = repeat(image_512, 'c h w -> (repeat c) h w', repeat=3)
        low_image = torch.from_numpy(low_image.astype(np.float32)).unsqueeze(0)
        low_image = repeat(low_image, 'c h w -> (repeat c) h w', repeat=3)
        label_512 = torch.from_numpy(label_512.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        binary_label_512 = torch.from_numpy(binary_label_512.astype(np.float32))
        low_binary_label = torch.from_numpy(low_binary_label.astype(np.float32))

        sample = {
            'image_512': image_512,
            'low_image': low_image,
            'label_512': label_512.long(),
            'low_res_label': low_res_label.long(),
            'binary_label_512': binary_label_512.long(),
            'low_binary_label': low_binary_label.long()  # 确保包含处理后的binary_label
        }
        return sample
    


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train_10pct":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        binary_label = (label > 0).astype(np.uint8)

        sample = {'image': image, 'label': label, 'binary_label': binary_label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample