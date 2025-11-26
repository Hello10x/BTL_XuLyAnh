import numpy as np
from .convolution import convolve
from .convolution import create_gaussian_kernel

def sobel_edge(img, kernel=None, ksize=3):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], float)

    sobel_y = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ], float)
    
    if kernel is not None:
        blur = convolve(img, kernel)
    elif ksize > 1:
        gauss = create_gaussian_kernel(ksize)
        blur = convolve(img, gauss)
    else:
        blur = img
        
    gx = convolve(blur, sobel_x)
    gy = convolve(blur, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)

    # Chuẩn hóa về 0-255
    if magnitude.max() != 0:
        magnitude = magnitude / magnitude.max() * 255
    else:
        magnitude = magnitude * 0  # tránh chia cho 0

    # Chuyển sang uint8
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return magnitude

