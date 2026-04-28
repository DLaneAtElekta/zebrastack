"""
Streamlit XR Viewer Component

A WebXR-enabled 3D viewer using A-Frame for viewing depth-mapped visualizations
in VR/AR headsets.
"""

from .xr_viewer import xr_viewer, export_mesh_to_glb

__all__ = ['xr_viewer', 'export_mesh_to_glb']
