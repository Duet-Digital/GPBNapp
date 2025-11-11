def generate_paint_by_numbers(
    image_file,
    num_colors=12,
    min_region_size=50,
    edge_thickness=1.0,
    contrast_factor=1.2,
    brightness_factor=1.05,
):
    import numpy as np
    import cv2
    from PIL import Image, ImageEnhance
    from scipy.cluster.vq import kmeans, vq
    from scipy import ndimage
    import io, zipfile
    import svgwrite
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------
    # Step 1: Load & enhance image
    # ---------------------------------------------------------
    image = Image.open(image_file).convert("RGB")
    enhancer_contrast = ImageEnhance.Contrast(image)
    enhancer_brightness = ImageEnhance.Brightness(enhancer_contrast.enhance(contrast_factor))
    img_enhanced = np.array(enhancer_brightness.enhance(brightness_factor))

    # ---------------------------------------------------------
    # Step 2: K-means color quantization
    # ---------------------------------------------------------
    pixels = img_enhanced.reshape(-1, 3).astype(float)
    centroids, _ = kmeans(pixels, num_colors)
    labels, _ = vq(pixels, centroids)
    quantized = centroids[labels].reshape(img_enhanced.shape).astype(np.uint8)

    # ---------------------------------------------------------
    # Step 3: Edge detection for outlines
    # ---------------------------------------------------------
    gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((int(edge_thickness), int(edge_thickness)), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # ---------------------------------------------------------
    # Step 4: Create base outline image
    # ---------------------------------------------------------
    outline_img = cv2.bitwise_not(edges)
    outline_rgb = cv2.cvtColor(outline_img, cv2.COLOR_GRAY2RGB)
    outline_pil = Image.fromarray(outline_rgb)

    # ---------------------------------------------------------
    # Step 5–7: Connected-region labeling and cleanup (fixed)
    # ---------------------------------------------------------
    labels_2d = labels.reshape(img_enhanced.shape[:2])

    region_labels = np.zeros_like(labels_2d, dtype=np.int32)
    region_counter = 0
    structure = np.ones((3, 3), dtype=np.int8)

    for color_idx in np.unique(labels_2d):
        mask = labels_2d == color_idx
        if not np.any(mask):
            continue
        labeled, num = ndimage.label(mask, structure)
        if num == 0:
            continue
        labeled[labeled > 0] += region_counter
        region_labels[mask] = labeled[mask]
        region_counter += num

    num_regions_init = region_counter

    # Step 6: Remove small regions
    sizes = ndimage.sum(
        np.ones_like(region_labels),
        region_labels,
        index=np.arange(1, num_regions_init + 1),
    )
    keep = sizes >= min_region_size
    keep_ids = np.flatnonzero(keep) + 1
    mask_keep = np.isin(region_labels, keep_ids)
    region_labels[~mask_keep] = 0

    # Relabel after cleanup
    region_labels, num_regions = ndimage.label(region_labels > 0, structure)

    # Step 7: Assign each region to nearest palette color
    region_to_color_idx = np.zeros(num_regions + 1, dtype=int)
    for rid in range(1, num_regions + 1):
        mask = region_labels == rid
        if mask.sum() == 0:
            continue
        avg_color = quantized[mask].mean(axis=0)
        region_to_color_idx[rid] = np.argmin(
            np.sum((centroids - avg_color) ** 2, axis=1)
        ) + 1

    # ---------------------------------------------------------
    # Step 8: Create reference image (color fill with outlines)
    # ---------------------------------------------------------
    ref_img = quantized.copy()
    ref_img[edges > 0] = 0

    # ---------------------------------------------------------
    # Step 9: Generate SVG outline with region numbers
    # ---------------------------------------------------------
    h, w = region_labels.shape
    dwg = svgwrite.Drawing(size=(w, h))
    for rid in range(1, num_regions + 1):
        mask = region_labels == rid
        if not np.any(mask):
            continue
        y, x = np.nonzero(mask)
        cy, cx = np.mean(y), np.mean(x)
        dwg.add(
            dwg.text(
                str(region_to_color_idx[rid]),
                insert=(cx, cy),
                fill="black",
                font_size="10px",
                text_anchor="middle",
            )
        )

    outline_buf = io.BytesIO()
    dwg.write(outline_buf)
    outline_buf.seek(0)

    # ---------------------------------------------------------
    # Step 10: Create palette preview
    # ---------------------------------------------------------
    fig, ax = plt.subplots(1, figsize=(6, 1))
    for i, c in enumerate(centroids):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c / 255.0))
        ax.text(i + 0.5, -0.3, str(i + 1), ha="center", va="top")
    ax.set_xlim(0, len(centroids))
    ax.set_ylim(0, 1)
    ax.axis("off")

    palette_buf = io.BytesIO()
    plt.savefig(palette_buf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    palette_buf.seek(0)

    # ---------------------------------------------------------
    # Step 11: Bundle outputs
    # ---------------------------------------------------------
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("outline.svg", outline_buf.getvalue())
        zf.writestr("palette.png", palette_buf.getvalue())

    zip_buf.seek(0)

    return zip_buf, io.BytesIO(ref_img.tobytes()), outline_buf, palette_buf, None, None