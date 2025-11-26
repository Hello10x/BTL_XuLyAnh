# Đặt hàm này trong một module mới (ví dụ: analysis.py) hoặc trong tệp canny.py
import numpy as np
from collections import deque # Dùng cho BFS (Hàng đợi)

def count_objects_cca(binary_img):
    """
    Đếm số lượng vật thể/thành phần liên thông (Connected Components Analysis - CCA)
    trên ảnh nhị phân (ảnh biên Canny) bằng thuật toán BFS.
    CHỈ SỬ DỤNG NUMPY (và deque từ collections).

    binary_img: Ảnh nhị phân (0 và 255), thường là đầu ra của Canny/threshold.
    Trả về: Số lượng vật thể.
    """
    # Đảm bảo ảnh là kiểu uint8, chuyển về nhị phân (0 hoặc 1)
    # Binary_img có thể có giá trị Strong=255, Weak=0, Background=0.
    # Ta chỉ quan tâm đến các điểm biên (255)
    img = (binary_img > 0).astype(np.uint8) 
    h, w = img.shape
    
    # Tạo mảng để theo dõi các pixel đã được thăm (visited)
    visited = np.zeros_like(img, dtype=bool)
    object_count = 0
    
    # Hướng di chuyển (8 lân cận): (dx, dy)
    # [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    # (ngang, dọc, chéo)
    directions = [(-1, -1), (-1, 0), (-1, 1), 
                  (0, -1),          (0, 1), 
                  (1, -1), (1, 0), (1, 1)]
    
    for i in range(h):
        for j in range(w):
            # Nếu pixel này là biên (1) và chưa được thăm
            if img[i, j] == 1 and not visited[i, j]:
                
                # Bắt đầu một vật thể mới
                object_count += 1
                
                # Khởi tạo Hàng đợi (Queue) cho BFS
                queue = deque([(i, j)])
                visited[i, j] = True
                
                while queue:
                    r, c = queue.popleft()
                    
                    # Kiểm tra 8 lân cận
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        
                        # Kiểm tra xem lân cận có nằm trong biên ảnh không
                        if 0 <= nr < h and 0 <= nc < w:
                            # Nếu lân cận là biên (1) và chưa được thăm
                            if img[nr, nc] == 1 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                                
    return object_count
# Cần import numpy, sử dụng các hàm convolve (tôi sẽ viết logic lân cận)

def erode(img, kernel_size=3):
    """
    Phép Xói mòn (Erosion) - Thay thế pixel bằng giá trị MIN trong cửa sổ.
    Kernel_size phải là số lẻ.
    """
    h, w = img.shape
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='constant', constant_values=255) # Đệm bằng giá trị biên 255 (trắng)
    out = np.zeros((h, w), dtype=img.dtype)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            # Xói mòn: giá trị đầu ra là MIN trong vùng kernel
            out[i, j] = np.min(region)
    return out

def dilate(img, kernel_size=3):
    """
    Phép Giãn nở (Dilation) - Thay thế pixel bằng giá trị MAX trong cửa sổ.
    Kernel_size phải là số lẻ.
    """
    h, w = img.shape
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0) # Đệm bằng giá trị nền 0 (đen)
    out = np.zeros((h, w), dtype=img.dtype)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            # Giãn nở: giá trị đầu ra là MAX trong vùng kernel
            out[i, j] = np.max(region)
    return out

def morphological_closing(img, kernel_size=3):
    """
    Phép Đóng = Giãn nở sau đó Xói mòn (Dilate -> Erode).
    Giúp lấp đầy các lỗ hổng nhỏ trong đường biên.
    """
    # 1. Giãn nở (Dilate)
    dilated = dilate(img, kernel_size)
    # 2. Xói mòn (Erode)
    closed = erode(dilated, kernel_size)
    return closed