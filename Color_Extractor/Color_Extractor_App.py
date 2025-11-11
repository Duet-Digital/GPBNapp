import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
import pandas as pd
from PIL import Image
import io
import colorsys

st.set_page_config(page_title="24-Color Palette Extractor", layout="wide")

# ========================================
# Title & Description
# ========================================
st.title("🎨 24-Color Palette Extractor")
st.markdown("""
Upload any image → get **exactly 24 dominant colors** with:
- Numbered swatches
- Hex, RGB values
- Usage percentage
- Downloadable PNG + CSV
""")

# ========================================
# Sidebar Controls
# ========================================
with st.sidebar:
    st.header("Settings")
    max_colors = st.slider("Max colors to extract", 4, 32, 24)
    resize_dim = st.slider("Resize for speed (px)", 100, 400, 150)
    min_usage = st.slider("Min % to show", 0.1, 5.0, 0.5, step=0.1)

# ========================================
# File Uploader
# ========================================
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # Resize for speed
    orig_w, orig_h = image.size
    scale = resize_dim / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Convert to numpy + reshape
    img_array = np.array(resized)
    h, w, _ = img_array.shape
    pixels = img_array.reshape(-1, 3).astype(np.float32)

    # ========================================
    # K-Means Clustering in LAB (perceptually uniform)
    # ========================================
    lab_pixels = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(lab_pixels)

    # Get cluster centers in RGB
    centers_lab = kmeans.cluster_centers_.astype(np.uint8)
    centers_rgb = cv2.cvtColor(centers_lab.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)

    # Count frequency
    counts = Counter(labels)
    total = len(labels)
    palette = []
    for i in range(max_colors):
        rgb = tuple(map(int, centers_rgb[i]))
        count = counts[i]
        percentage = (count / total) * 100
        if percentage >= min_usage:
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
            palette.append({
                'id': len(palette) + 1,
                'hex': hex_color,
                'rgb': rgb,
                'percent': percentage
            })

    # Sort by usage (descending)
    palette.sort(key=lambda x: x['percent'], reverse=True)
    final_palette = palette[:max_colors]

    # ========================================
    # Display Results
    # ========================================
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Top {len(final_palette)} Colors (≥ {min_usage}%)")
        df = pd.DataFrame([{
            'ID': c['id'],
            'Hex': c['hex'],
            'RGB': f"({c['rgb'][0]}, {c['rgb'][1]}, {c['rgb'][2]})",
            '%': f"{c['percent']:.2f}%"
        } for c in final_palette])
        st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("Color Swatch")
        swatch_size = 80
        swatch = np.ones((swatch_size, len(final_palette) * swatch_size, 3), dtype=np.uint8) * 255

        for idx, color in enumerate(final_palette):
            r, g, b = color['rgb']
            x = idx * swatch_size
            cv2.rectangle(swatch, (x, 0), (x + swatch_size, swatch_size), (b, g, r), -1)
            # Add number
            cv2.putText(swatch, str(color['id']), (x + 10, swatch_size - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

        swatch_rgb = cv2.cvtColor(swatch, cv2.COLOR_BGR2RGB)
        st.image(swatch_rgb, use_container_width=True)

    # ========================================
    # Download: PNG Swatch + CSV
    # ========================================
    buf = io.BytesIO()
    Image.fromarray(swatch_rgb).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    csv = df.to_csv(index=False).encode()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Swatch (PNG)",
            data=png_bytes,
            file_name="color_swatch.png",
            mime="image/png"
        )
    with col2:
        st.download_button(
            label="📄 Download Colors (CSV)",
            data=csv,
            file_name="palette.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload an image to extract its 24-color palette.")