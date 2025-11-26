import streamlit as st
import numpy as np
from PIL import Image
import io

from edge_processing.sobel import sobel_edge
from edge_processing.laplacian import laplacian_edge
from edge_processing.canny import canny_edge
from countobject import count_objects_cca
from collections import deque
from countobject import erode
from countobject import morphological_closing
from countobject import dilate


st.title("Xử lý biên ảnh bằng Canny, Sobel, Laplacian")

uploaded = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])

method = st.selectbox("Chọn phương pháp", ["Sobel", "Laplacian", "Canny"])
kernel = st.slider("Kernel size", 1, 7, 3, step=2)
count_objects = st.checkbox("Đếm vật thể", value=False)


# Chỉ hiển thị nếu chọn Canny
if method == "Canny":
    st.subheader("Tham số Canny")
    low_th = st.slider("Ngưỡng thấp (Low threshold)", 0, 255, 20)
    high_th = st.slider("Ngưỡng cao (High threshold)", 0, 255, 75)

if uploaded:
    img = Image.open(uploaded).convert("L")
    img_arr = np.array(img)

    # Gọi từng bộ xử lý
    if method == "Sobel":
    # kernel=None, ksize=slider
        result = sobel_edge(img_arr, kernel=None, ksize=kernel)
    elif method == "Laplacian":
        result = laplacian_edge(img_arr, kernel=None, ksize=kernel)
    else:
        result = canny_edge(img_arr, ksize=kernel, low=low_th, high=high_th)

    st.subheader("So sánh ảnh gốc và ảnh đã xử lý")
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded, caption="Ảnh gốc", use_container_width=True)

    with col2:
        st.image(result, caption=f"Ảnh xử lý ({method})", use_container_width=True)
    #đếm ảnh    
    if count_objects: 
        kernel_for_morph = 3
        result = morphological_closing(result, kernel_for_morph)
        object_count = count_objects_cca(result)
        st.success(f"**🔢 Số lượng Vật thể được Đếm (CCA):** **{object_count}**")

    # Nút tải ảnh
    st.subheader("Tải ảnh đã xử lý")

    result_img = Image.fromarray(result.astype(np.uint8))
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Tải xuống",
        data=byte_im,
        file_name=f"edge_{method.lower()}.png",
        mime="image/png"
    )
