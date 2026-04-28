"""
Streamlit XR Viewer Component

Provides WebXR-enabled 3D visualization using A-Frame, embedded in Streamlit
via an iframe component.
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
from pathlib import Path
import base64
import struct
import json
import io


def create_mesh_data(image_path: str, depth_path: str,
                     z_scale: float = 0.3,
                     downsample: int = 4,
                     use_hemisphere: bool = True,
                     depth_alpha: bool = False,
                     alpha_min: float = 0.2,
                     alpha_max: float = 1.0) -> dict:
    """
    Create mesh vertex and face data from image and depth map.

    Returns dict with vertices, faces, colors, and uvs for 3D mesh construction.

    Args:
        image_path: Path to the original image
        depth_path: Path to the depth map
        z_scale: Scale factor for depth displacement
        downsample: Factor to reduce resolution for performance
        use_hemisphere: Map onto hemisphere vs flat surface
        depth_alpha: If True, use depth to control alpha (closer=opaque, far=transparent)
        alpha_min: Minimum alpha value for farthest points (0-1)
        alpha_max: Maximum alpha value for closest points (0-1)
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

    # Flip for correct orientation
    depth_norm = np.flipud(depth_norm)
    img_arr = np.flipud(img_arr)

    height, width = depth_norm.shape

    if use_hemisphere:
        # Create hemisphere mapping
        u = np.linspace(-1, 1, width)
        v = np.linspace(-1, 1, height)
        uu, vv = np.meshgrid(u, v)

        r_uv = np.sqrt(uu**2 + vv**2)
        max_theta = np.pi / 2
        theta = np.clip(r_uv * max_theta, 0, max_theta)
        phi = np.arctan2(vv, uu)

        # Scale to reasonable VR size (1 meter base radius)
        base_radius = 0.5
        displacement = depth_norm * z_scale * base_radius
        radius = base_radius + displacement

        # Front hemisphere
        x = radius * np.cos(theta)
        y = radius * np.sin(theta) * np.sin(phi)  # Swap y/z for A-Frame coords
        z = radius * np.sin(theta) * np.cos(phi)
    else:
        # Flat surface - scale to ~1 meter
        scale = 1.0 / max(width, height)
        x_coords = np.arange(width) * scale - 0.5
        y_coords = np.arange(height) * scale - 0.5
        xx, yy = np.meshgrid(x_coords, y_coords)

        x = np.zeros_like(depth_norm)  # Flat on YZ plane
        y = yy
        z = xx
        # Add depth as x displacement
        x = depth_norm * z_scale * 0.5

    # Flatten vertices
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Create vertex colors (normalized 0-1)
    rgb = img_arr.reshape(-1, 3) / 255.0

    if depth_alpha:
        # Alpha based on depth: higher depth (closer) = more opaque
        # depth_norm is 0-1 where higher values are "closer" (more displaced)
        alpha = depth_norm.flatten() * (alpha_max - alpha_min) + alpha_min
        colors = np.column_stack([rgb, alpha]).astype(np.float32)
    else:
        colors = rgb.astype(np.float32)

    # Create UV coordinates
    u_coords = np.linspace(0, 1, width)
    v_coords = np.linspace(0, 1, height)
    uu, vv = np.meshgrid(u_coords, v_coords)
    uvs = np.stack([uu.flatten(), vv.flatten()], axis=1)

    # Create triangle faces
    faces = []
    face_depths = []  # Track depth for sorting

    for row in range(height - 1):
        for col in range(width - 1):
            top_left = row * width + col
            top_right = row * width + col + 1
            bottom_left = (row + 1) * width + col
            bottom_right = (row + 1) * width + col + 1

            # Two triangles per quad
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])

            if depth_alpha:
                # Calculate average depth for each triangle (for sorting)
                # x coordinate represents depth (higher = closer to camera)
                d1 = (vertices[top_left, 0] + vertices[bottom_left, 0] + vertices[top_right, 0]) / 3.0
                d2 = (vertices[top_right, 0] + vertices[bottom_left, 0] + vertices[bottom_right, 0]) / 3.0
                face_depths.extend([d1, d2])

    faces = np.array(faces, dtype=np.uint32)

    # Sort triangles back-to-front for proper alpha blending
    if depth_alpha and len(face_depths) > 0:
        face_depths = np.array(face_depths)
        sort_order = np.argsort(face_depths)  # Lower depth (farther) first
        faces = faces[sort_order]

    return {
        'vertices': vertices.astype(np.float32),
        'colors': colors,
        'uvs': uvs.astype(np.float32),
        'faces': faces,
        'width': width,
        'height': height,
        'has_alpha': depth_alpha
    }


def export_mesh_to_glb(image_path: str, depth_path: str,
                       output_path: str = None,
                       z_scale: float = 0.3,
                       downsample: int = 4,
                       use_hemisphere: bool = True,
                       depth_alpha: bool = False,
                       alpha_min: float = 0.2,
                       alpha_max: float = 1.0) -> bytes:
    """
    Export depth-mapped image as GLB (binary glTF) file.

    GLB is the standard format for WebXR 3D models.

    Args:
        image_path: Path to the original image
        depth_path: Path to the depth map
        output_path: Optional path to save the GLB file
        z_scale: Depth scale factor
        downsample: Resolution reduction factor
        use_hemisphere: Use hemisphere mapping vs flat surface
        depth_alpha: If True, use depth to control alpha transparency
        alpha_min: Minimum alpha for farthest points (0-1)
        alpha_max: Maximum alpha for closest points (0-1)

    Returns:
        GLB file as bytes
    """
    mesh_data = create_mesh_data(
        image_path, depth_path, z_scale, downsample, use_hemisphere,
        depth_alpha=depth_alpha, alpha_min=alpha_min, alpha_max=alpha_max
    )

    vertices = mesh_data['vertices']
    colors = mesh_data['colors']
    faces = mesh_data['faces']
    has_alpha = mesh_data['has_alpha']

    # Calculate bounds for accessor min/max
    v_min = vertices.min(axis=0).tolist()
    v_max = vertices.max(axis=0).tolist()

    # Build binary buffer
    # Layout: positions | colors | indices
    positions_bytes = vertices.tobytes()
    colors_bytes = colors.tobytes()
    indices_bytes = faces.flatten().astype(np.uint32).tobytes()

    # Pad to 4-byte alignment
    def pad_to_4(data):
        remainder = len(data) % 4
        if remainder:
            data += b'\x00' * (4 - remainder)
        return data

    positions_bytes = pad_to_4(positions_bytes)
    colors_bytes = pad_to_4(colors_bytes)
    indices_bytes = pad_to_4(indices_bytes)

    buffer_data = positions_bytes + colors_bytes + indices_bytes

    # Buffer views
    pos_offset = 0
    pos_length = len(positions_bytes)
    color_offset = pos_length
    color_length = len(colors_bytes)
    idx_offset = color_offset + color_length
    idx_length = len(indices_bytes)

    # Color type depends on whether we have alpha
    color_type = "VEC4" if has_alpha else "VEC3"
    num_color_components = 4 if has_alpha else 3

    # Build glTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "activation-atlas-xr"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "DepthSurface"}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "COLOR_0": 1
                },
                "indices": 2,
                "material": 0,
                "mode": 4  # TRIANGLES
            }],
            "name": "DepthMesh"
        }],
        "materials": [{
            "name": "DepthMaterial",
            "pbrMetallicRoughness": {
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8
            },
            "alphaMode": "BLEND" if has_alpha else "OPAQUE",
            # Use back-face culling for transparency (doubleSided=False)
            # This prevents artifacts from overlapping transparent faces
            "doubleSided": not has_alpha
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "min": v_min,
                "max": v_max
            },
            {
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": len(colors),
                "type": color_type
            },
            {
                "bufferView": 2,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(faces) * 3,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_length, "target": 34962},
            {"buffer": 0, "byteOffset": color_offset, "byteLength": color_length, "target": 34962},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_length, "target": 34963}
        ],
        "buffers": [{"byteLength": len(buffer_data)}]
    }

    # Encode JSON
    gltf_json = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    # Pad JSON to 4-byte alignment
    while len(gltf_json) % 4 != 0:
        gltf_json += b' '

    # Build GLB
    # Header: magic (4) + version (4) + length (4) = 12 bytes
    # JSON chunk: length (4) + type (4) + data
    # BIN chunk: length (4) + type (4) + data

    json_chunk_length = len(gltf_json)
    bin_chunk_length = len(buffer_data)

    total_length = 12 + 8 + json_chunk_length + 8 + bin_chunk_length

    glb = io.BytesIO()

    # Header
    glb.write(b'glTF')  # magic
    glb.write(struct.pack('<I', 2))  # version
    glb.write(struct.pack('<I', total_length))  # total length

    # JSON chunk
    glb.write(struct.pack('<I', json_chunk_length))  # chunk length
    glb.write(b'JSON')  # chunk type
    glb.write(gltf_json)

    # BIN chunk
    glb.write(struct.pack('<I', bin_chunk_length))  # chunk length
    glb.write(b'BIN\x00')  # chunk type
    glb.write(buffer_data)

    glb_bytes = glb.getvalue()

    if output_path:
        with open(output_path, 'wb') as f:
            f.write(glb_bytes)

    return glb_bytes


def get_aframe_html(glb_base64: str, title: str = "XR Viewer",
                    background_color: str = "#1a1a2e",
                    enable_vr: bool = True,
                    enable_ar: bool = True) -> str:
    """
    Generate A-Frame HTML for WebXR viewing.

    Args:
        glb_base64: Base64-encoded GLB model data
        title: Scene title
        background_color: Background color for the scene
        enable_vr: Enable VR mode button
        enable_ar: Enable AR mode button

    Returns:
        Complete HTML page with A-Frame scene
    """

    # Determine XR mode string
    xr_modes = []
    if enable_vr:
        xr_modes.append("vr")
    if enable_ar:
        xr_modes.append("ar")
    xr_mode_str = " ".join(xr_modes) if xr_modes else "vr"

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <meta name="description" content="WebXR 3D Depth Visualization">
    <script src="https://aframe.io/releases/1.5.0/aframe.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/aframe-orbit-controls@1.3.2/dist/aframe-orbit-controls.min.js"></script>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        .a-enter-vr-button, .a-enter-ar-button {{
            background: rgba(30, 30, 60, 0.9) !important;
            border: 2px solid #4a9eff !important;
            border-radius: 8px !important;
        }}
        .a-enter-vr-button:hover, .a-enter-ar-button:hover {{
            background: rgba(74, 158, 255, 0.8) !important;
        }}
        #info-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 14px;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
            z-index: 999;
        }}
    </style>
</head>
<body>
    <div id="info-panel">
        <strong>{title}</strong><br>
        <span id="xr-status">Checking XR support...</span><br>
        <small>Drag to rotate | Scroll to zoom</small>
    </div>

    <a-scene
        embedded
        webxr="requiredFeatures: hit-test,local-floor; optionalFeatures: hand-tracking,bounded-floor"
        xr-mode-ui="enabled: true; XRMode: {xr_mode_str}"
        renderer="antialias: true; colorManagement: true"
        background="color: {background_color}">

        <!-- Camera with orbit controls for desktop/mobile -->
        <a-entity id="camera-rig" position="0 0 2">
            <a-camera
                look-controls="enabled: false"
                wasd-controls="enabled: false"
                orbit-controls="
                    target: 0 0 0;
                    initialPosition: 0 0.5 1.5;
                    minDistance: 0.5;
                    maxDistance: 5;
                    enableDamping: true;
                    dampingFactor: 0.1;
                    rotateSpeed: 0.5;
                    zoomSpeed: 0.5;
                    autoRotate: false;
                    enablePan: true">
            </a-camera>
        </a-entity>

        <!-- Lighting -->
        <a-light type="ambient" color="#ffffff" intensity="0.6"></a-light>
        <a-light type="directional" color="#ffffff" intensity="0.8" position="1 2 1"></a-light>
        <a-light type="directional" color="#aaccff" intensity="0.4" position="-1 1 -1"></a-light>

        <!-- The 3D model -->
        <a-entity
            id="model-container"
            position="0 0 0"
            rotation="0 0 0"
            scale="1 1 1">
            <a-entity
                id="depth-model"
                gltf-model="url(data:model/gltf-binary;base64,{glb_base64})"
                position="0 0 0"
                material="vertexColors: true; side: double">
            </a-entity>
        </a-entity>

        <!-- Ground plane for AR reference -->
        <a-plane
            id="ground"
            rotation="-90 0 0"
            width="4"
            height="4"
            color="#1a1a2e"
            opacity="0.3"
            visible="false">
        </a-plane>

        <!-- Sky -->
        <a-sky color="{background_color}"></a-sky>

    </a-scene>

    <script>
        // Check XR support and update status
        async function checkXRSupport() {{
            const statusEl = document.getElementById('xr-status');
            let status = [];

            if (navigator.xr) {{
                try {{
                    const vrSupported = await navigator.xr.isSessionSupported('immersive-vr');
                    if (vrSupported) status.push('VR Ready');
                }} catch(e) {{}}

                try {{
                    const arSupported = await navigator.xr.isSessionSupported('immersive-ar');
                    if (arSupported) status.push('AR Ready');
                }} catch(e) {{}}
            }}

            if (status.length === 0) {{
                status.push('XR not available (use desktop/mobile controls)');
            }}

            statusEl.textContent = status.join(' | ');
        }}

        // Auto-rotate model slowly when not in XR
        AFRAME.registerComponent('auto-rotate', {{
            tick: function(time, delta) {{
                if (!this.el.sceneEl.is('vr-mode') && !this.el.sceneEl.is('ar-mode')) {{
                    this.el.object3D.rotation.y += 0.0005 * delta;
                }}
            }}
        }});

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            checkXRSupport();

            // Add auto-rotate to model container
            const modelContainer = document.getElementById('model-container');
            // modelContainer.setAttribute('auto-rotate', '');  // Uncomment for auto-rotation

            // Show ground in AR mode
            const scene = document.querySelector('a-scene');
            const ground = document.getElementById('ground');

            scene.addEventListener('enter-ar', () => {{
                ground.setAttribute('visible', 'true');
            }});

            scene.addEventListener('exit-ar', () => {{
                ground.setAttribute('visible', 'false');
            }});
        }});
    </script>
</body>
</html>'''

    return html


def xr_viewer(image_path: str, depth_path: str,
              z_scale: float = 0.3,
              downsample: int = 4,
              use_hemisphere: bool = True,
              height: int = 600,
              enable_vr: bool = True,
              enable_ar: bool = True,
              depth_alpha: bool = False,
              alpha_min: float = 0.2,
              alpha_max: float = 1.0) -> None:
    """
    Display an XR-ready 3D viewer in Streamlit.

    This creates an A-Frame WebXR scene that can be viewed in:
    - Desktop browsers (mouse/trackpad controls)
    - Mobile browsers (touch controls)
    - VR headsets (Quest, Vision Pro, etc.)
    - AR mode on supported devices

    Args:
        image_path: Path to the original image
        depth_path: Path to the depth map
        z_scale: Depth displacement scale (0.1-1.0)
        downsample: Resolution reduction for performance (1-8)
        use_hemisphere: Map onto hemisphere vs flat surface
        height: Viewer height in pixels
        enable_vr: Show VR mode button
        enable_ar: Show AR mode button
        depth_alpha: Use depth to control transparency (closer=opaque)
        alpha_min: Minimum alpha for farthest points
        alpha_max: Maximum alpha for closest points
    """
    # Generate GLB model
    glb_bytes = export_mesh_to_glb(
        image_path, depth_path,
        z_scale=z_scale,
        downsample=downsample,
        use_hemisphere=use_hemisphere,
        depth_alpha=depth_alpha,
        alpha_min=alpha_min,
        alpha_max=alpha_max
    )

    # Encode to base64
    glb_base64 = base64.b64encode(glb_bytes).decode('utf-8')

    # Get title from filename
    title = Path(image_path).stem

    # Generate HTML
    html = get_aframe_html(
        glb_base64=glb_base64,
        title=title,
        enable_vr=enable_vr,
        enable_ar=enable_ar
    )

    # Render in Streamlit
    components.html(
        html,
        height=height,
        scrolling=False
    )
