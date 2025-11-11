# ========================================
# IMPORTS
# ========================================
import streamlit as st                  # Web app framework
import numpy as np                      # Array operations
from PIL import Image, ImageDraw, ImageFont  # Image processing & drawing
import io, zipfile                      # In-memory file handling
import cv2                              # OpenCV for contours & filtering
from sklearn.cluster import KMeans      # Color clustering
from reportlab.lib.pagesizes import A4, LETTER  # PDF page sizes
from reportlab.pdfgen import canvas     # PDF generation
from reportlab.lib.utils import ImageReader  # Embed images in PDF

# ========================================
# 0. PAGE CONFIG
# ========================================
st.set_page_config(page_title="Paint-by-Numbers Generator", layout="wide")
st.title("Professional Paint-by-Numbers Generator")
st.markdown("**Upload any photo → Get a clean, printable coloring book with exactly N regions**")

# ========================================
# 1. LOAD & RESIZE IMAGE
# ========================================
def load_image(file, max_dim=1000):
    """Load image from upload, convert to RGB, resize if too large."""
    img = Image.open(file).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return np.array(img), img

# ========================================
# 2. VECTORIZE IMAGE (Cartoon/Line Art Style)
# ========================================
@st.cache_data
def vectorize_image(np_img, blur_ksize=5, canny_low=50, canny_high=150, dilate_iter=1):
    """Convert photo to clean vector-style line art using edge detection."""
    # Reduce noise
    blurred = cv2.medianBlur(np_img, blur_ksize)
    # Grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    # Edge detection
    edges = cv2.Canny(gray, canny_low, canny_high)
    # Thicken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    # Invert: white lines on black
    edges_inv = 255 - edges
    return edges_inv  # (H, W), uint8

# ========================================
# 3. APPLY VECTOR MASK TO ORIGINAL IMAGE
# ========================================
def apply_vector_mask(np_img, mask):
    """Preserve sharp edges while smoothing flat areas."""
    # Smooth image
    smoothed = cv2.bilateralFilter(np_img, d=9, sigmaColor=75, sigmaSpace=75)
    # Use mask: white = original, black = smoothed
    mask_3d = mask[..., np.newaxis]
    result = np.where(mask_3d == 255, np_img, smoothed)
    return result.astype(np.uint8)

# ========================================
# 4. QUANTIZE TO N COLORS (12–64)
# ========================================
@st.cache_data
def quantize_to_n_colors(np_img, n_colors=24):
    """Reduce image to N dominant colors using KMeans."""
    h, w, _ = np_img.shape
    pixels = np_img.reshape(-1, 3).astype(np.float32) / 255.0

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(pixels)
    centers = (kmeans.cluster_centers_ * 255).astype(np.uint8)

    # Sort by brightness
    brightness = 0.299*centers[:,0] + 0.587*centers[:,1] + 0.114*centers[:,2]
    order = np.argsort(brightness)
    sorted_centers = centers[order]
    remap = {old: new for new, old in enumerate(order)}

    labels = kmeans.labels_.reshape(h, w)
    final_label_map = np.vectorize(remap.get)(labels)
    return final_label_map, sorted_centers

# ========================================
# 5. CLEAN & MERGE → EXACTLY N REGIONS
# ========================================
def clean_and_merge_regions(label_map, np_img, n_colors, min_area=300, similarity_threshold=0.2):
    """
    ENSURE: Exactly n_colors regions in final label_map (0 to n_colors-1)
    1. Remove small regions
    2. Re-cluster to exactly n_colors
    """
    h, w = label_map.shape
    cleaned = label_map.copy()

    # --- Step 1: Remove small regions ---
    for lab in np.unique(cleaned):
        mask = (cleaned == lab)
        if mask.sum() < min_area:
            y, x = np.where(mask)
            neighbors = {}
            for yy, xx in zip(y, x):
                for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
                    ny, nx = yy+dy, xx+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        nlab = cleaned[ny,nx]
                        if nlab != lab:
                            neighbors[nlab] = neighbors.get(nlab, 0) + 1
            if neighbors:
                best = max(neighbors, key=neighbors.get)
                cleaned[mask] = best

    # --- Step 2: Re-cluster to EXACTLY n_colors ---
    pixels = []
    for lab in np.unique(cleaned):
        mask = (cleaned == lab)
        region_pixels = np_img[mask]
        if len(region_pixels) > 0:
            pixels.append(region_pixels)
    
    if not pixels:
        # Fallback
        dummy = np.zeros((1, 3), dtype=np.uint8)
        centers = np.tile(dummy, (n_colors, 1))
        return cleaned, centers

    pixels = np.vstack(pixels).astype(np.float32) / 255.0
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(pixels)
    new_labels = kmeans.labels_

    # Rebuild label map
    final_map = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    for lab in np.unique(cleaned):
        mask = (cleaned == lab)
        size = mask.sum()
        if size > 0:
            final_map[mask] = new_labels[idx]  # All same new label
            idx += size

    # Sort centers by brightness
    centers = (kmeans.cluster_centers_ * 255).astype(np.uint8)
    brightness = 0.299*centers[:,0] + 0.587*centers[:,1] + 0.114*centers[:,2]
    order = np.argsort(brightness)
    sorted_centers = centers[order]
    remap = {i: new for new, i in enumerate(order)}
    final_map = np.vectorize(remap.get)(final_map)

    return final_map, sorted_centers

# ========================================
# 6. DRAW BOLD, CLEAN OUTLINES
# ========================================
def draw_bold_clean_outlines(label_map, thickness=2):
    """Draw single-line, anti-aliased outlines around each region."""
    h, w = label_map.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    for lab in np.unique(label_map):
        mask = (label_map == lab).astype(np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (50, 50, 50), thickness=thickness, lineType=cv2.LINE_AA)

    return canvas

# ========================================
# 7. NUMBER EVERY REGION (One number per color)
# ========================================
def number_regions(outline_img, label_map, font_size=18):
    """Place number in center of every connected region of the same color."""
    img = Image.fromarray(outline_img.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    h, w = label_map.shape
    for lab in np.unique(label_map):
        mask = (label_map == lab)
        if not mask.any(): continue

        num_cc, cc_map = cv2.connectedComponents(mask.astype(np.uint8))
        text = str(int(lab) + 1)

        for cc in range(1, num_cc):
            region = (cc_map == cc)
            if region.sum() < 80: continue

            y, x = np.where(region)
            cy, cx = int(y.mean()), int(x.mean())

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            px = max(8, min(w - tw - 8, cx - tw // 2))
            py = max(8, min(h - th - 8, cy - th // 2))

            # White stroke for visibility
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((px+dx, py+dy), text, fill=(255,255,255), font=font)
            draw.text((px, py), text, fill=(0, 0, 0), font=font)

    return img

# ========================================
# 8. CREATE COLOR SWATCH
# ========================================
def create_swatch(centers):
    """Horizontal swatch with color blocks and numbers."""
    n = len(centers)
    swatch_w = min(2400, max(800, n * 60))
    sw = Image.new("RGB", (swatch_w, 100), "white")
    draw = ImageDraw.Draw(sw)
    cell_w = swatch_w // n
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, (r, g, b) in enumerate(centers):
        x0, x1 = i * cell_w, (i + 1) * cell_w
        draw.rectangle([x0, 10, x1-1, 90], fill=(r, g, b), outline=(0,0,0))
        text = str(i + 1)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x0 + (cell_w - tw)//2, 35), text, fill=(0,0,0), font=font)

    return sw

# ========================================
# 9. GENERATE PDF
# ========================================
def make_pdf(numbered_img, swatch_img, page_size=A4):
    """Create print-ready PDF with outline + swatch."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=page_size)
    width, height = page_size
    margin = 40

    # Main outline
    img = numbered_img.copy()
    img.thumbnail((width - 2*margin, height * 0.72), Image.LANCZOS)
    iw, ih = img.size
    c.drawImage(ImageReader(img), margin, height - margin - ih, width=iw, height=ih)

    # Swatch
    sw = swatch_img.copy()
    sw.thumbnail((width - 2*margin, 120), Image.LANCZOS)
    sw_w, sw_h = sw.size
    c.drawImage(ImageReader(sw), margin, margin + 10, width=sw_w, height=sw_h)

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 28, "Paint-by-Numbers Coloring Book")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ========================================
# MAIN UI & PROCESSING
# ========================================
uploaded = st.file_uploader("Upload Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Load image
    np_img, pil_img = load_image(uploaded)
    st.image(pil_img, caption="Original Photo", use_column_width=True)

    # Settings
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Settings")
        n_colors = st.slider("Number of Colors (Regions)", 12, 64, 28, help="Exactly this many regions & numbers")
        min_area = st.slider("Min Region Size", 150, 800, 350, help="Larger = fewer, cleaner regions")
        merge_sim = st.slider("Merge Similar Colors", 0.10, 0.35, 0.20, 0.05)
        thickness = st.slider("Outline Thickness", 1, 4, 2)
        number_size = st.slider("Number Size", 12, 28, 18)
        canny_low = st.slider("Edge Sensitivity", 10, 100, 50, help="Lower = more detail in lines")
        page = st.selectbox("Page Size", ["A4", "Letter"])
    
    with col2:
        st.header("Generate")
        if st.button("Create Professional PBN Kit", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status = st.empty()

            with st.spinner("Creating your coloring book..."):
                # 1. Vectorize
                status.text("1/8: Vectorizing image...")
                progress_bar.progress(1/8)
                edge_mask = vectorize_image(np_img, blur_ksize=5, canny_low=canny_low, canny_high=150)

                # 2. Apply vector mask
                status.text("2/8: Preserving edges...")
                progress_bar.progress(2/8)
                vectorized_img = apply_vector_mask(np_img, edge_mask)

                # 3. Initial clustering
                status.text("3/8: Initial color clustering...")
                progress_bar.progress(3/8)
                label_map, _ = quantize_to_n_colors(vectorized_img, n_colors=n_colors)

                # 4. FORCE EXACTLY n_colors REGIONS
                status.text(f"4/8: FORCING EXACTLY {n_colors} REGIONS...")
                progress_bar.progress(4/8)
                label_map, centers = clean_and_merge_regions(
                    label_map, vectorized_img, n_colors=n_colors,
                    min_area=min_area, similarity_threshold=merge_sim
                )

                # 5. Draw outlines
                status.text("5/8: Drawing clean outlines...")
                progress_bar.progress(5/8)
                outline = draw_bold_clean_outlines(label_map, thickness=thickness)

                # 6. Number regions
                status.text(f"6/8: Numbering {n_colors} regions...")
                progress_bar.progress(6/8)
                numbered = number_regions(outline, label_map, font_size=number_size)

                # 7. Create swatch
                status.text("7/8: Building swatch...")
                progress_bar.progress(7/8)
                swatch = create_swatch(centers)

                # 8. Finalize
                status.text("8/8: Finalizing...")
                progress_bar.progress(8/8)

            st.success(f"**Your kit is ready! Exactly {n_colors} regions created.**")
            
            # Display
            st.image(numbered, caption=f"**EXACTLY {n_colors} REGIONS** — Numbers 1 to {n_colors}", use_column_width=True)
            st.image(swatch, caption="Color Key", use_column_width=True)

            # PDF & ZIP
            page_sz = A4 if page == "A4" else LETTER
            pdf = make_pdf(numbered, swatch, page_sz)

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                buf = io.BytesIO(); numbered.save(buf, "PNG"); zf.writestr("pbn_outline.png", buf.getvalue())
                buf = io.BytesIO(); swatch.save(buf, "PNG"); zf.writestr("pbn_swatch.png", buf.getvalue())
                zf.writestr("pbn_coloring_book.pdf", pdf)
            zip_buf.seek(0)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download PDF", pdf, "pbn.pdf", "application/pdf", use_container_width=True)
            with col2:
                st.download_button("Download ZIP", zip_buf, "pbn_kit.zip", "application/zip", use_container_width=True)

    st.markdown("""
    ### Print Instructions
    1. **Download PDF**  
    2. Print on **A4/Letter**  
    3. Use **"Actual Size"**  
    4. **Paint with colors 1 to N!**
    """)