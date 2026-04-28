"""
Streamlit app for viewing visualizations with WebXR-enabled 3D depth surface rendering.

Features:
- Browse visualization images
- View corresponding depth maps
- Interactive 3D surface with texture mapping
- WebXR support for VR/AR headsets (Quest, Vision Pro, etc.)
"""

import streamlit as st
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Import the XR viewer component
from xr_viewer_component import xr_viewer, export_mesh_to_glb

# Set page config for wider layout
st.set_page_config(page_title="3D XR Visualization Viewer", layout="wide")


def find_image_pairs(base_path: str, recursive: bool = True) -> list:
    """
    Find all original images that have corresponding depth maps.

    Returns list of tuples: (original_path, depth_path)
    """
    base = Path(base_path)
    pattern = "**/*.png" if recursive else "*.png"

    pairs = []
    for img_path in base.glob(pattern):
        # Skip depth maps and yaml files
        if '_depth' in img_path.stem:
            continue

        # Check if depth map exists
        depth_path = img_path.parent / f"{img_path.stem}_depth{img_path.suffix}"
        if depth_path.exists():
            pairs.append((str(img_path), str(depth_path)))

    return pairs


def find_all_images(base_path: str, recursive: bool = True) -> list:
    """Find all original images (excluding depth maps)."""
    base = Path(base_path)
    pattern = "**/*.png" if recursive else "*.png"

    images = []
    for img_path in base.glob(pattern):
        if '_depth' not in img_path.stem:
            images.append(str(img_path))

    return images


# ============== Streamlit App ==============

st.title("🥽 3D XR Visualization Viewer")

st.markdown("""
**WebXR-enabled 3D viewer** - View neural network visualizations in VR/AR!

- **Desktop**: Drag to rotate, scroll to zoom
- **VR Headsets**: Click "Enter VR" button (Quest, Vision Pro, etc.)
- **AR Mode**: Available on supported mobile devices
""")

# Sidebar controls
st.sidebar.header("Settings")

base_path = st.sidebar.text_input("Base directory", value="screen_captures")
z_scale = st.sidebar.slider("Depth scale", 0.1, 1.0, 0.3, 0.05)
downsample = st.sidebar.slider("Quality (lower = faster)", 1, 8, 4, 1)
use_hemisphere = st.sidebar.checkbox("Hemisphere surface", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Transparency")
depth_alpha = st.sidebar.checkbox("Depth-based transparency", value=False,
                                   help="Make farther areas more transparent")
if depth_alpha:
    alpha_min = st.sidebar.slider("Min alpha (far)", 0.0, 1.0, 0.2, 0.05)
    alpha_max = st.sidebar.slider("Max alpha (close)", 0.0, 1.0, 1.0, 0.05)
else:
    alpha_min, alpha_max = 0.2, 1.0

st.sidebar.markdown("---")
st.sidebar.subheader("XR Options")
enable_vr = st.sidebar.checkbox("Enable VR mode", value=True)
enable_ar = st.sidebar.checkbox("Enable AR mode", value=True)
viewer_height = st.sidebar.slider("Viewer height", 400, 800, 600, 50)

# Find available images
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()


@st.cache_data
def get_image_pairs(path):
    return find_image_pairs(path)


@st.cache_data
def get_all_images(path):
    return find_all_images(path)


pairs = get_image_pairs(base_path)
all_images = get_all_images(base_path)

st.sidebar.write(f"Found {len(pairs)} images with depth maps")
st.sidebar.write(f"Found {len(all_images)} total images")

# Mode selection
mode = st.sidebar.radio("View mode", ["XR Viewer", "Side by side", "Export GLB"])

if mode == "XR Viewer":
    if not pairs:
        st.warning("No image pairs found. Run `python generate_depth_maps.py` first to generate depth maps.")
    else:
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("🎲 Random Image", use_container_width=True):
                st.session_state['current_pair'] = random.choice(pairs)

            # Image selector
            image_names = [Path(p[0]).stem for p in pairs]
            if 'current_pair' in st.session_state:
                current_idx = next((i for i, p in enumerate(pairs) if p[0] == st.session_state['current_pair'][0]), 0)
            else:
                current_idx = 0

            selected_name = st.selectbox("Or select image:", image_names, index=current_idx)
            selected_idx = image_names.index(selected_name)
            st.session_state['current_pair'] = pairs[selected_idx]

        if 'current_pair' not in st.session_state and pairs:
            st.session_state['current_pair'] = random.choice(pairs)

        if 'current_pair' in st.session_state:
            img_path, depth_path = st.session_state['current_pair']

            # Show thumbnails
            with col1:
                st.image(img_path, caption="Original", use_container_width=True)
                st.image(depth_path, caption="Depth Map", use_container_width=True)

            # XR Viewer
            with col2:
                st.subheader(f"🥽 {Path(img_path).stem}")

                with st.spinner("Generating XR scene..."):
                    xr_viewer(
                        img_path, depth_path,
                        z_scale=z_scale,
                        downsample=downsample,
                        use_hemisphere=use_hemisphere,
                        height=viewer_height,
                        enable_vr=enable_vr,
                        enable_ar=enable_ar,
                        depth_alpha=depth_alpha,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max
                    )

                st.info("💡 **Tip**: On Quest or Vision Pro, click 'Enter VR' to view in immersive mode!")

elif mode == "Side by side":
    if not pairs:
        st.warning("No image pairs found.")
    else:
        # Select image
        image_names = [Path(p[0]).stem for p in pairs]
        selected_name = st.selectbox("Select image", image_names)
        selected_idx = image_names.index(selected_name)
        img_path, depth_path = pairs[selected_idx]

        st.subheader(selected_name)

        # Layout: 2D images on left, XR viewer on right
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.image(img_path, caption="Original", use_container_width=True)
            st.image(depth_path, caption="Depth Map", use_container_width=True)

        with col_right:
            with st.spinner("Generating XR scene..."):
                xr_viewer(
                    img_path, depth_path,
                    z_scale=z_scale,
                    downsample=downsample,
                    use_hemisphere=use_hemisphere,
                    height=viewer_height,
                    enable_vr=enable_vr,
                    enable_ar=enable_ar,
                    depth_alpha=depth_alpha,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max
                )

elif mode == "Export GLB":
    st.subheader("Export 3D Model as GLB")
    st.markdown("""
    Export the depth-mapped visualization as a GLB file for use in:
    - Unity / Unreal Engine
    - Blender
    - Other WebXR apps
    - AR/VR platforms
    """)

    if not pairs:
        st.warning("No image pairs found.")
    else:
        image_names = [Path(p[0]).stem for p in pairs]
        selected_name = st.selectbox("Select image to export", image_names)
        selected_idx = image_names.index(selected_name)
        img_path, depth_path = pairs[selected_idx]

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_path, caption="Original", use_container_width=True)
        with col2:
            st.image(depth_path, caption="Depth Map", use_container_width=True)

        if st.button("📦 Generate GLB", use_container_width=True):
            with st.spinner("Generating GLB file..."):
                glb_bytes = export_mesh_to_glb(
                    img_path, depth_path,
                    z_scale=z_scale,
                    downsample=downsample,
                    use_hemisphere=use_hemisphere,
                    depth_alpha=depth_alpha,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max
                )

                st.download_button(
                    label="⬇️ Download GLB",
                    data=glb_bytes,
                    file_name=f"{selected_name}.glb",
                    mime="model/gltf-binary",
                    use_container_width=True
                )

                st.success(f"GLB generated! File size: {len(glb_bytes) / 1024:.1f} KB")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### XR Compatibility

**VR Headsets:**
- Meta Quest 2/3/Pro
- Apple Vision Pro
- HTC Vive / Valve Index

**AR Devices:**
- Android (Chrome)
- iOS (limited)

**Desktop:**
- Chrome/Edge/Firefox
""")
