import numpy as np

def convolve(img, kernel):
    """
    Thực hiện phép tích chập (Convolution) giữa ảnh và kernel.
    Sử dụng kỹ thuật zero-padding và duyệt ảnh bằng vòng lặp (non-vectorized).
    """
    img = img.astype(float)  # Đảm bảo ảnh là kiểu float để tính toán chính xác
    kh, kw = kernel.shape    # Lấy kích thước (cao, rộng) của kernel
    pad = kh // 2            # Tính toán độ rộng của đệm (padding) cần thiết

    # Đệm (Padding) ảnh
    # mode='reflect': Đệm bằng cách phản chiếu các giá trị pixel gần ranh giới
    padded = np.pad(img, pad, mode='reflect')
    h, w = img.shape         
    # Khởi tạo ma trận đầu ra với kích thước bằng ảnh gốc
    out = np.zeros((h, w), dtype=float)

    # Vòng lặp duyệt qua từng pixel trong ảnh gốc (hàng và cột)
    for i in range(h):
        for j in range(w):
            # Vùng này được căn chỉnh sao cho tâm kernel nằm tại (i, j) trong ảnh gốc
            region = padded[i:i + kh, j:j + kw]
            
            # sau đó tính tổng của kết quả. Đây chính là phép tính tích chập.
            out[i, j] = np.sum(region * kernel)

    return out


def create_gaussian_kernel(ksize, sigma=1.0):
    """
    Tạo kernel Gaussian 2D.
    - ksize: Kích thước kernel (phải là số lẻ, ví dụ: 3, 5, 7)
    - sigma: Độ lệch chuẩn (ảnh hưởng đến độ mờ)
    """
    # 1. Tạo mảng tọa độ 1D (ví dụ ksize=5: [-2., -1., 0., 1., 2.])
    ax = np.linspace(-(ksize-1)/2., (ksize-1)/2., ksize)
    
    # 2. Tạo ma trận tọa độ 2D (xx, yy) từ mảng 1D
    xx, yy = np.meshgrid(ax, ax)
    
    # 3. Tính toán giá trị kernel theo công thức Gaussian 2D: 
    # G(x, y) = e^(-(x^2 + y^2) / (2 * sigma^2))
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # 4. Chuẩn hóa kernel: Chia cho tổng của tất cả các phần tử để đảm bảo
    # tổng các trọng số bằng 1 (giữ nguyên độ sáng trung bình của ảnh).
    kernel /= np.sum(kernel)
    
    return kernel