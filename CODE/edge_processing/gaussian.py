import numpy as np
from .convolution import convolve

def gaussian_kernel(ksize, sigma=1.0):
    """
    Tạo kernel Gaussian 2D (Đã sửa lỗi np.meshmeshgrid thành np.meshgrid).
    - ksize: Kích thước kernel (phải là số lẻ).
    - sigma: Độ lệch chuẩn (kiểm soát độ mờ).
    """
    # Tạo mảng tọa độ 1D, lấy từ -(ksize//2) đến (ksize//2).
    # Ví dụ ksize=5: [-2., -1., 0., 1., 2.]
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    
    # Tạo ma trận tọa độ 2D (xx, yy) từ mảng 1D
    # **Đã sửa lỗi cú pháp từ meshmeshgrid thành meshgrid**
    xx, yy = np.meshgrid(ax, ax) 
    
    # Tính toán giá trị kernel theo công thức Gaussian 2D
    # G(x, y) = e^(-(x^2 + y^2) / (2 * sigma^2))
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma ** 2))
    
    # Chuẩn hóa kernel: Đảm bảo tổng các trọng số bằng 1
    # Giữ nguyên độ sáng trung bình của ảnh
    kernel /= np.sum(kernel)
    
    return kernel

def gaussian_blur(img, kernel):
    """
    Áp dụng bộ lọc Gaussian cho ảnh (Làm mờ/Giảm nhiễu).
    - img: Ảnh đầu vào (thang độ xám).
    - kernel: Kernel Gaussian đã được tạo.
    """
    # Thực hiện tích chập (Convolution) giữa ảnh và kernel Gaussian
    return convolve(img, kernel)
