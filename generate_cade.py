import imageio
import numpy as np
import matplotlib.pyplot as plt

def convert_eight_bit(img):
    img[img < 0] = 0
    img[img > 1] = 1
    img = (img * 255).astype(np.uint8)
    return img

def get_csdl_coeffs():
    np.random.seed()
    k0 = np.random.uniform(0, 1)
    k1 = np.random.uniform(0, 1)
    k2 = np.random.uniform(0, 1)
    return (k0, k1, k2)

def csdl(imgR, imgG, imgB):
    k0, k1, k2 = get_csdl_coeffs()
    img_s_RpG_denom = k0 + k1
    img_s_RpG = (k0 * imgR + k1 * imgG) / img_s_RpG_denom
    k0, k1, k2 = get_csdl_coeffs()
    img_s_GpB_denom = k1 + k2
    img_s_GpB = (k1 * imgG + k2 * imgB) / img_s_GpB_denom
    k0, k1, k2 = get_csdl_coeffs()
    img_s_RpB_denom = k0 + k2
    img_s_RpB = (k0 * imgR + k2 * imgB) / img_s_RpB_denom
    k0, k1, k2 = get_csdl_coeffs()
    img_s_RpGpB_denom = k0 + k1 + k2
    img_s_RpGpB = (k0 * imgR + k1 * imgG + k2 * imgB) / img_s_RpGpB_denom
    return [img_s_RpG, img_s_GpB, img_s_RpB, img_s_RpGpB]

def get_cade_coeffs():
    np.random.seed()
    k0 = np.random.uniform(0, 1)
    k1 = np.random.uniform(0, 1)
    return (k0, k1)

def cade(imgR, imgG, imgB):
    k0, k1 = get_cade_coeffs()
    imgBG_min = np.minimum(k0 * imgB, k1 * imgG)
    k0, k1 = get_cade_coeffs()
    imgGR_min = np.minimum(k0 * imgG, k1 * imgR)
    k0, k1 = get_cade_coeffs()
    imgBG_max = np.maximum(k0 * imgB, k1 * imgG)
    k0, k1 = get_cade_coeffs()
    imgGR_max = np.maximum(k0 * imgG, k1 * imgR)
    return [imgBG_min, imgGR_min, imgBG_max, imgGR_max]

def run():
    np.random.seed(45)
    rgb = imageio.v3.imread('rgb.png')/255

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    cade_imgs = [red, green, blue]

    imgs = csdl(red, green, blue)
    cade_imgs = cade_imgs + imgs
    imgs = cade(red, green, blue)
    cade_imgs = cade_imgs + imgs

    for i in range(len(cade_imgs)):
        imageio.v3.imwrite(f'cade_{i}.png', convert_eight_bit(cade_imgs[i]))

if __name__ == '__main__':
    run()
