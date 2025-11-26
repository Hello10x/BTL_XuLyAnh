import numpy as np
from .convolution import convolve
from .convolution import create_gaussian_kernel

def laplacian_edge(img, kernel=None, ksize=3):
    """
    img: ảnh grayscale
    kernel: kernel mờ tùy ý (np.array), nếu None sẽ dùng Gaussian kernel với ksize
    ksize: kích thước Gaussian kernel nếu kernel=None
    """
    # 1. Gaussian blur / hoặc kernel tùy ý
    if kernel is not None:
        blur = convolve(img, kernel)
    elif ksize > 1:
        gauss = create_gaussian_kernel(ksize)
        blur = convolve(img, gauss)
    else:
        blur = img

    # 2. Laplacian
    laplacian_kernel = np.array([[0,-1,0],
                             [-1,4,-1],
                             [0,-1,0]])

    lap = convolve(blur, laplacian_kernel)
    lap = np.abs(lap)

    # Chuẩn hóa về 0-255 và chuyển sang uint8
    if lap.max() != 0:
        lap = lap / lap.max() * 255
    else:
        lap = lap * 0

    lap = np.clip(lap, 0, 255).astype(np.uint8)

    return lap

