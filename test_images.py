import numpy as np
import imageio
import matplotlib.pyplot as plt

def degrade_img():
    img = np.ones((25, 25))
    for i in range(25):
        img[i][:] = 1 - (i*4)/100
    return img


def column_img():
    img = np.ones((25, 25))
    for i in range(25):
        for j in range(25):
            if j == 12:
                img[i][j] = 1 - (i*4)/100
    return img


def plus_img():
    img = np.ones((25, 25))
    for i in range(25):
        for j in range(25):
            if j == 12 or i == 12:
                img[i][j] = 0
    return img


def x_img():
    img = np.ones((25, 25))
    for i in range(25):
        for j in range(25):
            if j == i or i + j == 24:
                img[i][j] = 0
    return img


def diagonal_img():
    img = np.ones((25, 25))
    for i in range(25):
        for j in range(25):
            if i + j == 24:
                img[i][j] = 0
    return img


def inv_diagonal_img():
    img = np.ones((25, 25))
    for i in range(25):
        for j in range(25):
            if i == j:
                img[i][j] = 0
    return img


def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb, 0, 0.9999)


def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)


def to_gray(x):
    rgb = to_rgb(x)
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def load_emoji(index, path="data/emoji.png", size=40):
    im = imageio.imread(path)
    emoji = np.array(im[:, index*size:(index+1)*size].astype(np.float32))
    emoji /= 255.0
    gray_emoji = to_gray(emoji)
    return gray_emoji


def print_img(img):
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()

def grayscale_to_rgb(img):
    blue_channel = np.array(img*255, dtype = 'uint8')
    red_channel = np.array(img*255, dtype = 'uint8')
    green_channel = np.array(img*255, dtype = 'uint8')
    
    return np.stack((red_channel, blue_channel, green_channel), axis=2)