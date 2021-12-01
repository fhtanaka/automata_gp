import numpy as np
import matplotlib.pyplot as plt
import imageio

WHITE = 255
BLACK = 0

def degrade_img(size):
    img = np.full((size, size), WHITE)
    for i in range(size):
        img[i][:] = WHITE - i * int(WHITE/size)
    return img


def one_color_img(size, color=122):
    img = np.full((size, size), WHITE)
    for i in range(size):
        img[i][:] = color
    return img


def degrade_column_img(size):
    img = np.full((size, size), WHITE)
    for i in range(size):
        for j in range(size):
            if j == int(size/2):
                img[i][j] = WHITE - i * int(WHITE/size)
    return img


def plus_img(size):
    img = np.full((size, size), WHITE)
    for i in range(size):
        for j in range(size):
            if j == 12 or i == 12:
                img[i][j] = BLACK
    return img


def x_img(size):
    img = np.full((size, size), WHITE)
    for i in range(size):
        for j in range(size):
            if j == i or i + j == size-1:
                img[i][j] = BLACK
    return img


def diagonal_img(size):
    img = np.full((size, size), WHITE)
    for i in range(size):
        for j in range(size):
            if i + j == size-1:
                img[i][j] = BLACK
    return img


def inv_diagonal_img(size):
    img = np.full((size, size), WHITE)
    for i in range(size):
        for j in range(size):
            if i == j:
                img[i][j] = BLACK
    return img


def print_img(img, scale = 4):
    plt.figure(figsize=(scale, scale))
    plt.imshow(img, cmap='gray', vmin=BLACK, vmax=WHITE)
    plt.show()

