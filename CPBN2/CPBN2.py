# ========================================
# Processing (OPTIMIZED REGION DETECTION)
# ========================================
outline_file = None
palette_file = None

if outline_file is not None and palette_file is not None:
    # your processing code here

    # -------------------------------
    # 2. Load palette
    # -------------------------------
    palette_df = pd.read_csv(palette_file)
    required_cols = ['Hex']
    if not all(col in palette_df.columns for col in required_cols):
        st.error(f"CSV must contain: {', '.join(required_cols)}")
        st.stop()

    palette_rgb = []
    for hex_color in palette_df['Hex']:
        hex_clean = hex_color.lstrip('#')
        rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        palette_rgb.append(rgb)
    palette_rgb = np.array(palette_rgb, dtype=np.uint8)
    palette_lab = cv2.cvtColor(palette_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)

    # -------------------------------
    # 3. Sidebar: Detection Settings
    # -------------------------------
    # -------------------------------
# 3. Sidebar: Detection Settings
# -------------------------------
with st.sidebar:
    st.header("Region Detection Settings")
    min_area = st.slider("Min region area (px²)", 20, 500, 100)
    close_kernel = st.slider("Morph close kernel", 3, 21, 7, step=2)
    adaptive_block = st.slider("Adaptive block size", 11, 101, 51, step=2)
    adaptive_c = st.slider("Adaptive C", 2, 20, 8)
    use_ocr = st.checkbox("Use OCR to read numbers (pytesseract)", value=True)

    st.header("Numbering Style")
    grid_step   = st.slider("Number spacing (px)", 30, 120, 60)
    font_scale  = st.slider("Font scale", 0.3, 1.0, 0.45, step=0.05)
    min_dist    = st.slider("Min distance from edge (px)", 3, 15, 6)

    # -------------------------------
    # 4. Preprocess: Adaptive threshold + close gaps
    # -------------------------------
    if gray.mean() > 127:
        # Dark lines on light bg → invert
        gray = 255 - gray
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # -------------------------------
    # 5. Connected Components
    # -------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    st.info(f"Found {num_labels-1} connected regions (incl. background)")

    # Filter by area
    valid_labels = []
    for i in range(1, num_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_labels.append(i)

    st.success(f"Kept {len(valid_labels)} regions after area filter")

    # -------------------------------
    # 6. Extract region color + OCR number
    # -------------------------------
    region_data = []
    mask_all = np.zeros(gray.shape, dtype=np.uint8)

    for label in valid_labels:
        # Create mask for this component
        mask = (labels == label).astype(np.uint8) * 255
        mask_all |= mask

        # Erode slightly to avoid edge/number bleed
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(mask, erode_kernel, iterations=2)

        # Sample inner pixels
        pixels = outline_cv[inner_mask > 0]
        if len(pixels) == 0:
            continue
        median_color_bgr = np.median(pixels, axis=0).astype(np.uint8)
        median_color_rgb = median_color_bgr[[2, 1, 0]]  # BGR → RGB

        # Convert to LAB
        lab_color = cv2.cvtColor(median_color_rgb.reshape(1,1,3), cv2.COLOR_RGB2LAB).flatten()

        # OCR: extract number
        number = None
        if use_ocr:
            try:
                import pytesseract
                # Crop region with padding
                x, y, w, h = cv2.boundingRect(mask)
                pad = 10
                crop = gray[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]
                # Preprocess for OCR
                crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                crop = cv2.copyMakeBorder(crop, 20,20,20,20, cv2.BORDER_CONSTANT, value=255)
                text = pytesseract.image_to_string(
                    crop, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
                )
                text = ''.join(filter(str.isdigit, text))
                if text:
                    number = int(text)
            except Exception as e:
                st.warning(f"OCR failed: {e}")

        # Fallback: use centroid
        cx, cy = map(int, centroids[label])

        region_data.append({
            'label': label,
            'mask': mask,
            'inner_mask': inner_mask,
            'median_rgb': median_color_rgb,
            'lab': lab_color,
            'center': (cx, cy),
            'number': number,
            'area': stats[label, cv2.CC_STAT_AREA]
        })

    if not region_data:
        st.error("No valid regions after filtering.")
        st.stop()

    # -------------------------------
    # 7. Match to Palette (Delta-E 2000)
    # -------------------------------
    from skimage.color import deltaE_ciede2000

    assignments = []
    for region in region_data:
        deltas = [deltaE_ciede2000(region['lab'], palette_lab[i]) for i in range(len(palette_lab))]
        best_idx = int(np.argmin(deltas))
        color_id = palette_df.iloc[best_idx]['ID'] if 'ID' in palette_df.columns else best_idx + 1
        assignments.append({
            'region': region,
            'palette_idx': best_idx,
            'color_id': color_id,
            'rgb': palette_rgb[best_idx]
        })

    # -------------------------------
# 8. Render Final Image – SMALLER & FREQUENT NUMBERS
# -------------------------------
with st.sidebar:
    st.header("Numbering Style")
    grid_step   = st.slider("Number spacing (px)", 30, 120, 60)      # distance between numbers
    font_scale  = st.slider("Font scale", 0.3, 1.0, 0.45, step=0.05) # tiny numbers
    min_dist    = st.slider("Min distance from edge (px)", 3, 15, 6)

result = outline_cv.copy()

# Lighten base a little so numbers stay visible
result = cv2.addWeighted(result, 0.25, np.full_like(result, 255), 0.75, 0)

for assign in assignments:
    region   = assign['region']
    b, g, r  = assign['rgb']
    color_bgr = (int(b), int(g), int(r))

    # ---- Fill the whole region ----
    cv2.fillPoly(result,
                 [cv2.findContours(region['mask'], cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)[0]],
                 color_bgr)

    # ---- Create inner mask (erode to stay away from edges) ----
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2*min_dist+1, 2*min_dist+1))
    inner_mask = cv2.erode(region['mask'], erode_k, iterations=1)

    # ---- Grid of numbers inside the inner mask ----
    ys, xs = np.where(inner_mask > 0)
    if len(xs) == 0:
        continue

    # Sub‑sample to a regular grid
    step = grid_step
    for y in range(ys.min(), ys.max() + step, step):
        for x in range(xs.min(), xs.max() + step, step):
            if inner_mask[y, x] == 0:
                continue
            # Draw tiny number
            cv2.putText(result, str(assign['color_id']),
                        (x - 4, y + 6),                     # slight offset for centering
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), 1, cv2.LINE_AA)

# Convert back to RGB for Streamlit
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)