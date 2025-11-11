import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io, zipfile, cv2
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Paint-by-Numbers Generator", layout="wide")
st.title("🎨 Professional Paint-by-Numbers Generator")
st.markdown("**Upload any photo → Get a clean, printable coloring book (like commercial PBN kits)**")

# ======================
# 1. Load & Resize
# ======================
def load_image(file, max_dim=1000):
    img = Image.open(file).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return np.array(img), img

# ======================
# 2. Quantize to N Colors (12–64)
# ======================
@st.cache_data
def quantize_to_n_colors(np_img, n_colors=24):
    h, w, _ = np_img.shape
    pixels = np_img.reshape(-1, 3).astype(np.float32) / 255.0

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(pixels)
    centers = (kmeans.cluster_centers_ * 255).astype(np.uint8)

    # Sort by brightness (dark to light)
    brightness = 0.299*centers[:,0] + 0.587*centers[:,1] + 0.114*centers[:,2]
    order = np.argsort(brightness)
    sorted_centers = centers[order]
    remap = {old: new for new, old in enumerate(order)}

    labels = kmeans.labels_.reshape(h, w)
    final_label_map = np.vectorize(remap.get)(labels)
    return final_label_map, sorted_centers

# ======================
# 3. AGGRESSIVE CLEANING + MERGING (Key to clean outlines)
# ======================
def clean_and_merge_regions(label_map, np_img, min_area=300, similarity_threshold=0.2):
    """
    Remove tiny regions + merge similar neighboring regions for clean outlines
    """
    h, w = label_map.shape
    cleaned = label_map.copy()

    # Step 1: Remove small regions
    for lab in np.unique(label_map):
        mask = (label_map == lab)
        if mask.sum() < min_area:
            y, x = np.where(mask)
            neighbors = {}
            for yy, xx in zip(y, x):
                for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
                    ny, nx = yy+dy, xx+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        nlab = label_map[ny,nx]
                        if nlab != lab:
                            neighbors[nlab] = neighbors.get(nlab, 0) + 1
            if neighbors:
                best = max(neighbors, key=neighbors.get)
                cleaned[mask] = best

    # Step 2: Get average colors for each label
    centers = []
    for lab in np.unique(cleaned):
        mask = (cleaned == lab)
        if mask.sum() > 0:
            centers.append(np_img[mask].mean(axis=0).astype(np.float32))
        else:
            centers.append(np.array([128,128,128], dtype=np.float32))
    centers = np.array(centers)

    # Step 3: Iteratively merge similar neighboring regions
    iterations = 0
    max_iterations = 10
    while iterations < max_iterations:
        changed = False
        new_map = cleaned.copy()
        for lab in np.unique(cleaned):
            mask = (cleaned == lab)
            y, x = np.where(mask)
            if len(y) == 0: continue
            
            for yy, xx in zip(y, x):
                for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
                    ny, nx = yy+dy, xx+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        nlab = cleaned[ny,nx]
                        if nlab != lab and nlab < len(centers):
                            # Normalized RGB distance
                            dist = np.linalg.norm(centers[lab] - centers[nlab]) / (255 * np.sqrt(3))
                            if dist < similarity_threshold:
                                new_map[mask] = nlab
                                changed = True
                                break
                if changed: break
            if changed: break
        cleaned = new_map
        if not changed: break
        iterations += 1

    return cleaned

# ======================
# 4. EDGE-AWARE SMOOTHING
# ======================
def smooth_color_regions(np_img, label_map):
    """Smooth within regions, preserve edges"""
    h, w = label_map.shape
    smoothed = np_img.copy().astype(np.uint8)

    for lab in np.unique(label_map):
        mask = (label_map == lab)
        if mask.sum() == 0: continue
        
        # Bilateral filter preserves edges while smoothing
        region_smoothed = cv2.bilateralFilter(
            np_img, d=9, sigmaColor=75, sigmaSpace=75
        )
        smoothed[mask] = region_smoothed[mask]

    return smoothed

# ======================
# 5. CLEAN, BOLD OUTLINES
# ======================
def draw_bold_clean_outlines(label_map, thickness=2):
    h, w = label_map.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # Slightly dilate masks for bolder, cleaner lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    for lab in np.unique(label_map):
        mask = (label_map == lab).astype(np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (50, 50, 50), thickness=thickness, lineType=cv2.LINE_AA)

    return canvas

# ======================
# 6. Number EVERY Region
# ======================
def number_regions(outline_img, label_map, font_size=18):
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
            if region.sum() < 80: continue  # Skip tiny areas

            y, x = np.where(region)
            cy, cx = int(y.mean()), int(x.mean())

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            px = max(8, min(w - tw - 8, cx - tw // 2))
            py = max(8, min(h - th - 8, cy - th // 2))

            # Bold text with outline for better visibility
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((px+dx, py+dy), text, fill=(255,255,255), font=font)
            draw.text((px, py), text, fill=(0, 0, 0), font=font)

    return img

# ======================
# 7. Create Swatch
# ======================
def create_swatch(centers):
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

# ======================
# 8. PDF Layout
# ======================
def make_pdf(numbered_img, swatch_img, page_size=A4):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=page_size)
    width, height = page_size
    margin = 40

    # Numbered outline (main image)
    img = numbered_img.copy()
    img.thumbnail((width - 2*margin, height * 0.72), Image.LANCZOS)
    iw, ih = img.size
    c.drawImage(ImageReader(img), margin, height - margin - ih, width=iw, height=ih)

    # Swatch (bottom)
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

# ======================
# MAIN UI
# ======================
uploaded = st.file_uploader("📁 Upload Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    np_img, pil_img = load_image(uploaded)
    st.image(pil_img, caption="Original Photo", use_column_width=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("⚙️ Settings")
        n_colors = st.slider("Number of Colors", 12, 64, 28, help="More colors = more detail")
        min_area = st.slider("Min Region Size", 150, 800, 350, help="Larger = cleaner outlines")
        merge_sim = st.slider("Merge Similar", 0.10, 0.35, 0.20, 0.05, help="Lower = more distinct colors")
        thickness = st.slider("Outline Thickness", 1, 4, 2)
        number_size = st.slider("Number Size", 12, 28, 18)
        page = st.selectbox("Page Size", ["A4", "Letter"])
    
    with col2:
        st.header("🎨 Generate")
        if st.button("✨ Create Professional PBN Kit", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Processing your coloring book..."):
                status_text.text("🔍 Step 1/7: Finding colors...")
                progress_bar.progress(1/7)
                label_map, centers = quantize_to_n_colors(np_img, n_colors=n_colors)

                status_text.text("🧹 Step 2/7: Cleaning & merging regions...")
                progress_bar.progress(2/7)
                label_map = clean_and_merge_regions(label_map, np_img, min_area=min_area, similarity_threshold=merge_sim)

                status_text.text("🎨 Step 3/7: Smoothing shapes...")
                progress_bar.progress(3/7)
                smoothed_img = smooth_color_regions(np_img, label_map)

                status_text.text("📐 Step 4/7: Re-quantizing smoothed image...")
                progress_bar.progress(4/7)
                label_map, centers = quantize_to_n_colors(smoothed_img, n_colors=n_colors)

                status_text.text("✏️ Step 5/7: Drawing clean outlines...")
                progress_bar.progress(5/7)
                outline = draw_bold_clean_outlines(label_map, thickness=thickness)

                status_text.text("🔢 Step 6/7: Numbering every region...")
                progress_bar.progress(6/7)
                numbered = number_regions(outline, label_map, font_size=number_size)

                status_text.text("🎨 Step 7/7: Creating color swatch...")
                progress_bar.progress(7/7)
                swatch = create_swatch(centers)

            st.success("✅ **Your professional Paint-by-Numbers kit is ready!**")
            
            # Display results
            st.image(numbered, caption=f"✨ Numbered Outline ({len(np.unique(label_map))} Clean Regions)", use_column_width=True)
            st.image(swatch, caption="🎨 Color Key", use_column_width=True)

            # Create downloads
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
                st.download_button(
                    "📄 Download PDF (Print Ready!)", 
                    pdf, 
                    "pbn_coloring_book.pdf", 
                    "application/pdf",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "📦 Download Full ZIP", 
                    zip_buf, 
                    "pbn_complete_kit.zip", 
                    "application/zip",
                    use_container_width=True
                )

    st.markdown("""
    ### 🖨️ **Print Instructions**
    1. Download **PDF** 
    2. Print on **A4/Letter** paper
    3. Select **"Actual Size"** (100% scale)
    4. **Paint away!** 🎨
    """)

# Footer
st.markdown("---")
st.markdown("*Powered by advanced computer vision & K-Means clustering*")