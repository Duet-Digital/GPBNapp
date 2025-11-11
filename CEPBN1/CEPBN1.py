import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
from collections import defaultdict
import colorsys

st.set_page_config(page_title="Paint-by-Number Numbering", layout="wide")

st.title("🎨 Paint-by-Number Auto-Numbering")
st.markdown("""
**Upload your outline + palette → Get fully numbered paint-by-number!**

1. Outline image (black lines, white background)
2. Color palette CSV (from previous app)
3. ✅ Auto-numbers ALL sections
""")

# ========================================
# File Uploads
# ========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Outline")
    outline_file = st.file_uploader("Outline Image", type=["png", "jpg", "jpeg"], key="outline")

with col2:
    st.subheader("2. Upload Palette")
    palette_file = st.file_uploader("Palette CSV", type=["csv"], key="palette")

# ========================================
# Process Button
# ========================================
if st.button("🎯 GENERATE NUMBERED OUTLINE", type="primary"):
    if outline_file is not None and palette_file is not None:
        # Load outline
        outline_pil = Image.open(outline_file).convert("RGB")
        outline_cv = cv2.cvtColor(np.array(outline_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(outline_cv, cv2.COLOR_BGR2GRAY)
        
        # Load palette
        palette_df = pd.read_csv(palette_file)
        palette_rgb = []
        for _, row in palette_df.iterrows():
            rgb = eval(row['RGB'].replace('(', '').replace(')', ''))
            palette_rgb.append(np.array(rgb))
        palette_rgb = np.array(palette_rgb)
        
        st.success(f"✅ Loaded outline ({outline_pil.size}) + {len(palette_df)} colors")
        
        # ========================================
        # STEP 1: Find ALL white regions (blobs)
        # ========================================
        # Clean up outline - threshold to pure B&W
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        # Invert (white regions become black for floodfill)
        regions = cv2.bitwise_not(binary)
        
        # Find all contours (regions to number)
        contours, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        st.info(f"Found {len(contours)} regions to number")
        
        # ========================================
        # STEP 2: Sample color from each region
        # ========================================
        region_colors = []
        for i, cnt in enumerate(contours):
            # Get region mask
            mask = np.zeros(regions.shape, np.uint8)
            cv2.fillPoly(mask, [cnt], 255)
            
            # Sample multiple points from region
            points = np.column_stack(np.where(mask > 0))
            if len(points) > 0:
                samples = points[np.random.choice(len(points), min(50, len(points)), replace=False)]
                region_color = np.mean(outline_cv[samples[:,0], samples[:,1]], axis=0).astype(np.uint8)
                region_colors.append(region_color)
            else:
                region_colors.append([255, 255, 255])  # Default white
        
        # ========================================
        # STEP 3: Match regions to closest palette color
        # ========================================
        numbered_outline = outline_cv.copy()
        color_assignments = []
        
        for i, region_color in enumerate(region_colors):
            # Convert to LAB for perceptual distance
            region_lab = cv2.cvtColor(region_color.reshape(1,1,3), cv2.COLOR_RGB2LAB)[0,0]
            palette_lab = cv2.cvtColor(palette_rgb, cv2.COLOR_RGB2LAB)
            
            # Find closest palette color
            distances = np.linalg.norm(palette_lab - region_lab, axis=1)
            closest_idx = np.argmin(distances)
            
            color_assignments.append({
                'region': i+1,
                'color_id': closest_idx + 1,
                'distance': distances[closest_idx]
            })
            
            # Place number in region center
            M = cv2.moments(contours[i])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw number (black with white outline for visibility)
                cv2.putText(numbered_outline, str(closest_idx + 1), 
                           (cx-12, cy+12), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.2, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(numbered_outline, str(closest_idx + 1), 
                           (cx-12, cy+12), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.2, (0, 0, 0), 2, cv2.LINE_AA)
        
        # ========================================
        # Display Results
        # ========================================
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Outline")
            st.image(outline_pil, use_column_width=True)
        
        with col2:
            st.subheader("✅ Fully Numbered!")
            numbered_rgb = cv2.cvtColor(numbered_outline, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(numbered_rgb), use_column_width=True)
        
        with col3:
            st.subheader("Color Legend")
            legend_size = 40
            legend = np.ones((len(palette_df)*legend_size, 200, 3), np.uint8) * 255
            
            for i, (_, row) in enumerate(palette_df.iterrows()):
                rgb = eval(row['RGB'].replace('(', '').replace(')', ''))
                cv2.rectangle(legend, (10, i*legend_size), 
                             (10+legend_size, (i+1)*legend_size), rgb, -1)
                cv2.putText(legend, f"{row['ID']}", (70, i*legend_size + 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                cv2.putText(legend, row['Hex'], (120, i*legend_size + 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
            st.image(cv2.cvtColor(legend, cv2.COLOR_RGB2BGR), use_column_width=True)
        
        # ========================================
        # Downloads
        # ========================================
        st.subheader("📥 Downloads")
        col1, col2 = st.columns(2)
        
        # Numbered PNG
        buf = io.BytesIO()
        Image.fromarray(numbered_rgb).save(buf, format="PNG")
        with col1:
            st.download_button(
                label="🎨 Numbered Outline (PNG)",
                data=buf.getvalue(),
                file_name="paint_by_number.png",
                mime="image/png"
            )
        
        # Updated CSV with region counts
        assignment_df = pd.DataFrame(color_assignments)
        counts = assignment_df['color_id'].value_counts().sort_index()
        final_csv = palette_df.merge(
            pd.DataFrame({'color_id': counts.index, 'regions': counts.values}),
            left_on='ID', right_on='color_id', how='left'
        ).fillna(0)
        
        with col2:
            csv_buf = io.StringIO()
            final_csv.to_csv(csv_buf, index=False)
            st.download_button(
                label="📊 Updated Palette (CSV)",
                data=csv_buf.getvalue().encode(),
                file_name="numbered_palette.csv",
                mime="text/csv"
            )
        
        st.balloons()

# ========================================
# Instructions
# ========================================
if outline_file is None or palette_file is None:
    st.markdown("""
    ### 🚀 Quick Start
    1. **Outline**: Use the previous app to generate black-on-white outline
    2. **Palette**: Use the palette extractor → download CSV  
    3. **Click GENERATE** → Get fully numbered paint-by-number!
    """)