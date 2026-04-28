"""
Streamlit app for viewing visualizations with 3D depth surface rendering.

Features:
- Browse visualization images
- View corresponding depth maps
- Interactive 3D surface with texture mapping using Plotly
"""

import streamlit as st
import random
from pathlib import Path
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Set page config for wider layout
st.set_page_config(page_title="3D Visualization Viewer", layout="wide")


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


def create_3d_surface(image_path: str, depth_path: str,
                      z_scale: float = 0.3,
                      downsample: int = 4) -> go.Figure:
    """
    Create a 3D surface plot with the image texture mapped onto the depth surface.

    Args:
        image_path: Path to the original image
        depth_path: Path to the depth map
        z_scale: Scale factor for depth (height)
        downsample: Factor to reduce resolution for performance
    """
    # Load images
    img = Image.open(image_path).convert('RGB')
    depth = Image.open(depth_path).convert('L')

    # Resize depth to match image if needed
    if img.size != depth.size:
        depth = depth.resize(img.size, Image.Resampling.BILINEAR)

    # Downsample for performance
    new_size = (img.size[0] // downsample, img.size[1] // downsample)
    img_small = img.resize(new_size, Image.Resampling.LANCZOS)
    depth_small = depth.resize(new_size, Image.Resampling.BILINEAR)

    # Convert to numpy
    img_arr = np.array(img_small)
    depth_arr = np.array(depth_small).astype(float)

    # Normalize depth and apply scale
    depth_norm = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8)
    z = depth_norm * z_scale * max(new_size)

    # Create meshgrid for x, y coordinates
    height, width = depth_arr.shape
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # Create RGB surfacecolor
    # Normalize to 0-1 for plotly
    surfacecolor = img_arr.astype(float) / 255.0

    # Create custom colorscale from the image
    # We'll use the image directly as surfacecolor
    # Convert RGB to a single value for colorscale mapping
    # Use grayscale intensity for the colorscale reference
    intensity = np.mean(surfacecolor, axis=2)

    # Build the figure
    fig = go.Figure()

    # Add the surface
    fig.add_trace(go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=intensity,
        colorscale=[
            [0, f'rgb({int(img_arr[0,0,0])},{int(img_arr[0,0,1])},{int(img_arr[0,0,2])})'],
            [1, f'rgb({int(img_arr[-1,-1,0])},{int(img_arr[-1,-1,1])},{int(img_arr[-1,-1,2])})']
        ],
        showscale=False,
        lighting=dict(
            ambient=0.8,
            diffuse=0.5,
            specular=0.1,
            roughness=0.5
        ),
        # Use the actual image as texture via customdata
        hovertemplate="x: %{x}<br>y: %{y}<br>depth: %{z:.2f}<extra></extra>"
    ))

    # For true texture mapping, we need to use image as surfacecolor
    # Create a proper texture-mapped surface
    fig.data[0].update(
        surfacecolor=img_arr[:,:,0] * 0.299 + img_arr[:,:,1] * 0.587 + img_arr[:,:,2] * 0.114,
        colorscale=create_image_colorscale(img_arr),
    )

    # Update layout
    fig.update_layout(
        title=dict(text=Path(image_path).stem, x=0.5),
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            aspectmode='data',
            camera=dict(
                eye=dict(x=0, y=-1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )

    return fig


def create_image_colorscale(img_arr: np.ndarray) -> list:
    """
    Create a colorscale from image pixels for approximate texture mapping.

    This samples colors from the image to create a gradient colorscale.
    """
    # Flatten and sample colors
    h, w, _ = img_arr.shape

    # Sample along diagonal for color variety
    n_samples = 256
    colorscale = []

    for i in range(n_samples):
        t = i / (n_samples - 1)
        y_idx = int(t * (h - 1))
        x_idx = int(t * (w - 1))
        r, g, b = img_arr[y_idx, x_idx]
        colorscale.append([t, f'rgb({r},{g},{b})'])

    return colorscale


def create_texture_surface(image_path: str, depth_path: str,
                          z_scale: float = 0.3,
                          downsample: int = 4,
                          use_hemisphere: bool = True,
                          depth_alpha: bool = False,
                          depth_darken: bool = False,
                          effect_min: float = 0.2,
                          effect_max: float = 1.0) -> go.Figure:
    """
    Create a 3D surface with proper texture mapping using Mesh3d with vertexcolor.

    This creates a triangulated mesh where each vertex gets its actual RGB color
    from the corresponding pixel in the image, providing true texture mapping.

    Args:
        image_path: Path to the original image
        depth_path: Path to the depth map
        z_scale: Scale factor for depth displacement
        downsample: Factor to reduce resolution for performance
        use_hemisphere: If True, map onto a hemisphere with depth as displacement
        depth_alpha: If True, use depth to control alpha (closer=opaque, far=transparent)
        depth_darken: If True, darken far areas (no sorting needed, works with camera rotation)
        effect_min: Minimum effect value for farthest points (0-1)
        effect_max: Maximum effect value for closest points (0-1)
    """
    # Load images
    img = Image.open(image_path).convert('RGB')
    depth = Image.open(depth_path).convert('L')

    # Resize depth to match image if needed
    if img.size != depth.size:
        depth = depth.resize(img.size, Image.Resampling.BILINEAR)

    # Downsample for performance
    new_size = (img.size[0] // downsample, img.size[1] // downsample)
    img_small = img.resize(new_size, Image.Resampling.LANCZOS)
    depth_small = depth.resize(new_size, Image.Resampling.BILINEAR)

    # Convert to numpy
    img_arr = np.array(img_small)
    depth_arr = np.array(depth_small).astype(float)

    # Normalize depth to 0-1 range
    depth_norm = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8)

    # Flip Y axis for correct orientation
    depth_norm = np.flipud(depth_norm)
    img_arr = np.flipud(img_arr)

    height, width = depth_norm.shape

    if use_hemisphere:
        # Create two mirrored hemispheres to form an enclosed sphere
        # Front hemisphere: image with depth displacement outward
        # Back hemisphere: mirror image with depth displacement outward (opposite direction)

        # u, v are normalized coordinates from -1 to 1
        u = np.linspace(-1, 1, width)
        v = np.linspace(-1, 1, height)
        uu, vv = np.meshgrid(u, v)

        # Calculate radius from center for hemisphere mapping
        r_uv = np.sqrt(uu**2 + vv**2)

        # Map to hemisphere: center -> pole, edges -> equator
        max_theta = np.pi / 2  # Full hemisphere (90 degrees)
        theta = r_uv * max_theta
        theta = np.clip(theta, 0, max_theta)

        # Azimuthal angle around the axis
        phi = np.arctan2(vv, uu)

        # Base radius for the sphere
        base_radius = max(width, height) * 0.4

        # Depth displacement
        displacement = depth_norm * z_scale * base_radius
        radius = base_radius + displacement

        # === Front hemisphere (bulging toward +x) ===
        x_front = radius * np.cos(theta)
        y_front = radius * np.sin(theta) * np.cos(phi)
        z_front = radius * np.sin(theta) * np.sin(phi)

        # === Back hemisphere (bulging toward -x, mirrored) ===
        x_back = -radius * np.cos(theta)
        y_back = radius * np.sin(theta) * np.cos(phi)  # Same y
        z_back = radius * np.sin(theta) * np.sin(phi)  # Same z

        # Combine both hemispheres
        x_flat = np.concatenate([x_front.flatten(), x_back.flatten()])
        y_flat = np.concatenate([y_front.flatten(), y_back.flatten()])
        z_flat = np.concatenate([z_front.flatten(), z_back.flatten()])

        # Flag to handle doubled colors/triangles later
        img_arr_doubled = True
    else:
        # Original flat surface with depth as z
        z = depth_norm * z_scale * max(new_size)

        x_coords = np.arange(width)
        y_coords = np.arange(height)
        xx, yy = np.meshgrid(x_coords, y_coords)

        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z_flat = z.flatten()
        img_arr_doubled = False

    # Create vertex colors from image (flatten RGB values)
    r_flat = img_arr[:, :, 0].flatten().astype(float)
    g_flat = img_arr[:, :, 1].flatten().astype(float)
    b_flat = img_arr[:, :, 2].flatten().astype(float)

    # Create color strings for each vertex (with optional effects based on depth)
    if depth_alpha:
        # Alpha based on depth: higher depth (closer) = more opaque
        alpha_flat = depth_norm.flatten() * (effect_max - effect_min) + effect_min
        single_colors = [f'rgba({int(r)},{int(g)},{int(b)},{a:.3f})' for r, g, b, a in zip(r_flat, g_flat, b_flat, alpha_flat)]
    elif depth_darken:
        # Darken based on depth: higher depth (closer) = brighter, lower = darker
        # This doesn't require sorting and works correctly with camera rotation
        brightness_flat = depth_norm.flatten() * (effect_max - effect_min) + effect_min
        r_dark = (r_flat * brightness_flat).astype(int)
        g_dark = (g_flat * brightness_flat).astype(int)
        b_dark = (b_flat * brightness_flat).astype(int)
        single_colors = [f'rgb({r},{g},{b})' for r, g, b in zip(r_dark, g_dark, b_dark)]
    else:
        single_colors = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in zip(r_flat, g_flat, b_flat)]

    # Double the colors if we have two hemispheres
    if img_arr_doubled:
        vertex_colors = single_colors + single_colors
    else:
        vertex_colors = single_colors

    # Create triangle indices
    # Each quad (4 adjacent pixels) becomes 2 triangles
    i_list = []  # First vertex of each triangle
    j_list = []  # Second vertex
    k_list = []  # Third vertex

    n_vertices_per_hemisphere = height * width

    for row in range(height - 1):
        for col in range(width - 1):
            # Index of the four corners of this quad
            top_left = row * width + col
            top_right = row * width + col + 1
            bottom_left = (row + 1) * width + col
            bottom_right = (row + 1) * width + col + 1

            # Front hemisphere triangles (winding order for front-facing)
            i_list.append(top_left)
            j_list.append(bottom_left)
            k_list.append(top_right)

            i_list.append(top_right)
            j_list.append(bottom_left)
            k_list.append(bottom_right)

            # Back hemisphere triangles (offset by n_vertices, reverse winding)
            if img_arr_doubled:
                back_offset = n_vertices_per_hemisphere
                i_list.append(top_left + back_offset)
                j_list.append(top_right + back_offset)
                k_list.append(bottom_left + back_offset)

                i_list.append(top_right + back_offset)
                j_list.append(bottom_right + back_offset)
                k_list.append(bottom_left + back_offset)

    # Sort triangles by depth for proper alpha blending (back-to-front)
    if depth_alpha:
        # Calculate centroid depth for each triangle
        i_arr = np.array(i_list)
        j_arr = np.array(j_list)
        k_arr = np.array(k_list)

        # Get average x position (depth) for each triangle
        # For hemisphere, x is the depth axis (positive = closer to camera)
        tri_depth = (x_flat[i_arr] + x_flat[j_arr] + x_flat[k_arr]) / 3.0

        # Sort triangles back-to-front (lower x first, then higher x on top)
        sort_order = np.argsort(tri_depth)

        i_list = i_arr[sort_order].tolist()
        j_list = j_arr[sort_order].tolist()
        k_list = k_arr[sort_order].tolist()

    # Create the figure with Mesh3d
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        i=i_list,
        j=j_list,
        k=k_list,
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting=dict(
            ambient=0.8,
            diffuse=0.4,
            specular=0.1,
            roughness=0.6
        ),
        lightposition=dict(x=0, y=0, z=1000),
        hoverinfo='skip'
    ))

    # Adjust camera based on surface type
    if use_hemisphere:
        camera = dict(
            eye=dict(x=2.0, y=1.0, z=0.8),  # View from front-right-top
            up=dict(x=0, y=0, z=1)
        )
    else:
        camera = dict(
            eye=dict(x=0, y=-1.8, z=1.0),
            up=dict(x=0, y=0, z=1)
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=camera,
            bgcolor='rgb(20,20,20)'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        paper_bgcolor='rgb(20,20,20)'
    )

    return fig


def build_rgb_colorscale(img_arr: np.ndarray) -> list:
    """Build colorscale that maps intensity to actual image colors."""
    # Flatten image and sort by intensity
    h, w, _ = img_arr.shape
    pixels = img_arr.reshape(-1, 3)
    intensity = pixels[:, 0] * 0.299 + pixels[:, 1] * 0.587 + pixels[:, 2] * 0.114

    # Sort by intensity and sample
    sorted_idx = np.argsort(intensity)
    n_samples = 256
    sample_idx = np.linspace(0, len(sorted_idx) - 1, n_samples).astype(int)

    colorscale = []
    for i, idx in enumerate(sample_idx):
        t = i / (n_samples - 1)
        pixel_idx = sorted_idx[idx]
        r, g, b = pixels[pixel_idx]
        colorscale.append([t, f'rgb({int(r)},{int(g)},{int(b)})'])

    return colorscale


# ============== Streamlit App ==============

st.title("🏔️ 3D Visualization Viewer")

# Sidebar controls
st.sidebar.header("Settings")

base_path = st.sidebar.text_input("Base directory", value="screen_captures")
z_scale = st.sidebar.slider("Depth scale", 0.1, 1.0, 0.3, 0.05)
downsample = st.sidebar.slider("Quality (lower = faster)", 1, 8, 4, 1)
use_hemisphere = st.sidebar.checkbox("Hemisphere surface", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Depth Effects")
depth_effect = st.sidebar.radio("Depth effect", ["None", "Transparency", "Darken"],
                                 help="None: solid colors, Transparency: alpha blend (static sort), Darken: dim far areas")
depth_alpha = depth_effect == "Transparency"
depth_darken = depth_effect == "Darken"
if depth_effect != "None":
    effect_min = st.sidebar.slider("Min effect (far)", 0.0, 1.0, 0.2, 0.05)
    effect_max = st.sidebar.slider("Max effect (close)", 0.0, 1.0, 1.0, 0.05)
else:
    effect_min, effect_max = 1.0, 1.0

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
mode = st.sidebar.radio("View mode", ["Random with depth", "Browse all", "Side by side"])

if mode == "Random with depth":
    if not pairs:
        st.warning("No image pairs found. Run `python generate_depth_maps.py` first to generate depth maps.")
    else:
        if st.button("🎲 Random Image"):
            st.session_state['current_pair'] = random.choice(pairs)

        if 'current_pair' not in st.session_state and pairs:
            st.session_state['current_pair'] = random.choice(pairs)

        if 'current_pair' in st.session_state:
            img_path, depth_path = st.session_state['current_pair']

            st.subheader(Path(img_path).stem)

            # Layout: left 1/3 for 2D images stacked, right 2/3 for 3D
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.image(img_path, caption="Original", use_container_width=True)
                st.image(depth_path, caption="Depth Map", use_container_width=True)

            with col_right:
                with st.spinner("Generating 3D surface..."):
                    fig = create_texture_surface(
                        img_path, depth_path, z_scale, downsample, use_hemisphere,
                        depth_alpha, depth_darken, effect_min, effect_max)
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)

elif mode == "Browse all":
    if not all_images:
        st.warning("No images found in the specified directory.")
    else:
        # Pagination
        images_per_page = 12
        total_pages = (len(all_images) - 1) // images_per_page + 1
        page = st.sidebar.number_input("Page", 1, total_pages, 1)

        start_idx = (page - 1) * images_per_page
        end_idx = min(start_idx + images_per_page, len(all_images))

        st.write(f"Showing {start_idx + 1}-{end_idx} of {len(all_images)} images")

        # Display grid
        cols = st.columns(4)
        for i, img_path in enumerate(all_images[start_idx:end_idx]):
            with cols[i % 4]:
                st.image(img_path, caption=Path(img_path).stem[:30], use_container_width=True)

                # Check if depth exists
                depth_path = Path(img_path).parent / f"{Path(img_path).stem}_depth.png"
                if depth_path.exists():
                    if st.button("View 3D", key=f"3d_{i}"):
                        st.session_state['selected_for_3d'] = (img_path, str(depth_path))

        # Show selected 3D view
        if 'selected_for_3d' in st.session_state:
            img_path, depth_path = st.session_state['selected_for_3d']
            st.subheader(f"3D View: {Path(img_path).stem}")
            fig = create_texture_surface(
                img_path, depth_path, z_scale, downsample, use_hemisphere,
                depth_alpha, depth_darken, effect_min, effect_max)
            st.plotly_chart(fig, use_container_width=True)

elif mode == "Side by side":
    if not pairs:
        st.warning("No image pairs found.")
    else:
        # Select image
        image_names = [Path(p[0]).stem for p in pairs]
        selected_name = st.selectbox("Select image", image_names)
        selected_idx = image_names.index(selected_name)
        img_path, depth_path = pairs[selected_idx]

        # Layout: left 1/3 for 2D images stacked, right 2/3 for 3D
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.image(img_path, caption="Original", use_container_width=True)
            st.image(depth_path, caption="Depth Map", use_container_width=True)

        with col_right:
            with st.spinner("Generating 3D surface..."):
                fig = create_texture_surface(
                    img_path, depth_path, z_scale, downsample, use_hemisphere,
                    depth_alpha, depth_darken, effect_min, effect_max)
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
