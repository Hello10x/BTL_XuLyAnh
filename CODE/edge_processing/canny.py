import numpy as np
from .convolution import convolve, create_gaussian_kernel

# =========================================
# 1. NON-MAX SUPPRESSION
# =========================================
def non_max_suppression(grad, angle):
    """
    Áp dụng non-maximum suppression để mảnh hóa biên.
    - grad: magnitude của gradient
    - angle: hướng gradient (radian)
    """
    h, w = grad.shape
    out = np.zeros((h, w))

    # Đổi góc về [0..180)
    angle = (angle * 180.0 / np.pi) % 180  

    for i in range(1, h-1):
        for j in range(1, w-1):

            # Mặc định so sánh 2 pixel q và r theo hướng gradient
            q = r = 0

            # 0 degrees (ngang)
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = grad[i, j+1]
                r = grad[i, j-1]

            # 45 degrees
            elif 22.5 <= angle[i,j] < 67.5:
                q = grad[i+1, j-1]
                r = grad[i-1, j+1]

            # 90 degrees (dọc)
            elif 67.5 <= angle[i,j] < 112.5:
                q = grad[i+1, j]
                r = grad[i-1, j]

            # 135 degrees
            elif 112.5 <= angle[i,j] < 157.5:
                q = grad[i-1, j-1]
                r = grad[i+1, j+1]

            # Giữ nếu là cực đại
            if grad[i,j] >= q and grad[i,j] >= r:
                out[i,j] = grad[i,j]

    return out


# =========================================
# 2. DOUBLE THRESHOLD + HYSTERESIS
# =========================================
def double_threshold_hysteresis(img, low, high):
    """
    Áp dụng:
    - Double threshold
    - Hysteresis để mở rộng biên mạnh

    Low < Weak < High < Strong
    """
    strong = 255
    weak = 50

    res = np.zeros_like(img, dtype=np.uint8)

    # Phân loại pixel
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # Hysteresis: weak → strong nếu cạnh mạnh kế cận
    h, w = res.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if res[i, j] == weak:
                if strong in res[i-1:i+2, j-1:j+2]:
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    return res


# =========================================
# 3. FULL CANNY PIPELINE
# =========================================
def canny_edge(img, ksize=5, low=50, high=150):
    """
    Pipeline Canny đầy đủ:
    - ksize: kernel Gaussian
    - low, high: ngưỡng cho double threshold
    """

    # 1. Gaussian blur
    kernel = create_gaussian_kernel(ksize)
    blur = convolve(img, kernel)

    # 2. Gradient Sobel
    gx = convolve(blur, np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
    gy = convolve(blur, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))

    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = (magnitude / magnitude.max()) * 255
    angle = np.arctan2(gy, gx)

    # 3. Non-max suppression
    nms = non_max_suppression(magnitude, angle)

    # 4 + 5. Double threshold & hysteresis
    edges = double_threshold_hysteresis(nms, low, high)

    return edges
