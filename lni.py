import numpy as np
import imageio

import torch
import torch.nn.functional as F

def convert_eight_bit(img):
    img[img < 0] = 0
    img[img > 1] = 1
    img = (img * 255).astype(np.uint8)
    return img

def calc(img, block_size):
    block_size_half = block_size//2

    img = F.pad(img, (block_size_half, block_size_half, block_size_half, block_size_half), 'replicate')

    mean = integral_sum(img, block_size)/(block_size ** 2)
    stddev = integral_stddev(img, mean, block_size)

    img = img[:, :, block_size_half:-block_size_half, block_size_half:-block_size_half]
    img[stddev > 0] = (img[stddev > 0] - mean[stddev > 0])/stddev[stddev > 0]
    img[img > 1] = 1
    img[img < -1] = -1

    img = img * 0.5 + 0.5

    return img

def integral_sum(image, win_size):
    win_size_red = win_size - 1

    integral_image = image.cumsum(axis=2, dtype=torch.float64).cumsum(axis=3, dtype=torch.float64)
    _, _, height, width = integral_image.shape
    sum = image.new_zeros((1, 1, height - win_size_red, width - win_size_red))
    sum[:, :, 0, 0] = integral_image[:, :, win_size_red, win_size_red]
    sum[:, :, 0, 1:] = integral_image[:, :, win_size_red, win_size:width] - integral_image[:, :, win_size_red, :(width - win_size)]
    sum[:, :, 1:, 0] = integral_image[:, :, win_size:height, win_size_red] - integral_image[:, :, :(height - win_size), win_size_red]
    sum[:, :, 1:, 1:] = integral_image[:, :, win_size:height, win_size:width] - integral_image[:, :, win_size:height, :(width - win_size)] - integral_image[:, :, :(height - win_size), win_size:width] + integral_image[:, :, :(height - win_size), :(width - win_size)]
    return sum

def integral_stddev(image, mean, win_size):
    win_size_red = win_size - 1

    sq_image = image ** 2
    integral_image = sq_image.cumsum(axis=2, dtype=torch.float64).cumsum(axis=3, dtype=torch.float64)
    sq_mean = (win_size ** 2) * (mean ** 2)

    _, _, height, width = integral_image.shape

    stddev = image.new_zeros((1, 1, height - win_size_red, width - win_size_red))
    stddev[:, :, 0, 0] = integral_image[:, :, win_size_red, win_size_red] - sq_mean[:, :, 0, 0]
    stddev[:, :, 0, 1:] = integral_image[:, :, win_size_red, win_size:width] - integral_image[:, :, win_size_red, :(width - win_size)] - sq_mean[:, :, 0, 1:]
    stddev[:, :, 1:, 0] = integral_image[:, :, win_size:height, win_size_red] - integral_image[:, :, :(height - win_size), win_size_red] - sq_mean[:, :, 1:, 0]
    stddev[:, :, 1:, 1:] = integral_image[:, :, win_size:height, win_size:width] - integral_image[:, :, win_size:height, :(width - win_size)] - integral_image[:, :, :(height - win_size), win_size:width] + integral_image[:, :, :(height - win_size), :(width - win_size)] - sq_mean[:, :, 1:, 1:]
    stddev = stddev

    stddev[stddev < 0] = 0
    stddev = torch.sqrt(stddev)/win_size
    return stddev
    


def run():
    img = imageio.v3.imread('rgb.png')/255
    img = img[:, :, 0]
    
    img = torch.from_numpy(img)[None, None, ...]
    
    lni = calc(img, block_size=3)
    
    lni = lni.numpy()[0, 0]
    imageio.v3.imwrite(f'lni.png', convert_eight_bit(lni))

if __name__ == '__main__':
    run()
