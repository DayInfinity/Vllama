import os
import sys
import signal
import numpy as np

import open3d as o3d
import trimesh
import pyrender

from plyfile import PlyData
from scipy.special import expit


# ------------------------------------------------------
# Graceful keyboard interrupt handler
# ------------------------------------------------------
def _handle_interrupt(sig, frame):
    print("\n[INFO] Viewer closed by user (Ctrl+C)")
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_interrupt)


# ------------------------------------------------------
# Gaussian Splatting PLY Viewer (Open3D)
# ------------------------------------------------------
def _view_gaussian_ply(ply_path: str):
    """
    Visualize Gaussian Splatting PLY using SH DC → RGB
    """
    print("[INFO] Detected Gaussian Splatting PLY")

    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    # XYZ
    points = np.vstack([v["x"], v["y"], v["z"]]).T

    # SH DC color → RGB
    f_dc = np.vstack([
        v["f_dc_0"],
        v["f_dc_1"],
        v["f_dc_2"]
    ]).T

    colors = expit(f_dc)
    colors = np.clip(colors, 0.0, 1.0)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("[INFO] Launching Open3D viewer...")
    o3d.visualization.draw_geometries([pcd])


# ------------------------------------------------------
# Normal PLY Viewer (Open3D)
# ------------------------------------------------------
def _view_standard_ply(ply_path: str):
    """
    Visualize standard PLY (XYZ or XYZ+RGB)
    """
    print("[INFO] Detected standard PLY")

    pcd = o3d.io.read_point_cloud(ply_path)

    if not pcd.has_colors():
        print("[WARN] PLY has no colors, displaying geometry only")

    o3d.visualization.draw_geometries([pcd])


# ------------------------------------------------------
# Mesh Viewer (GLB / OBJ / STL / FBX)
# ------------------------------------------------------
def _view_mesh(mesh_path: str):
    """
    Visualize mesh formats using trimesh + pyrender
    """
    print("[INFO] Detected mesh format")

    tm_obj = trimesh.load(mesh_path, force="scene")

    if isinstance(tm_obj, trimesh.Scene):
        scene = pyrender.Scene.from_trimesh_scene(tm_obj)
    else:
        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(tm_obj)
        scene.add(mesh)

    print("[INFO] Launching Pyrender viewer...")
    pyrender.Viewer(scene, use_raymond_lighting=True)


# ------------------------------------------------------
# Public API: Unified Viewer
# ------------------------------------------------------
def view_3d_model(model_path: str):
    """
    Automatically detect and display a 3D model.

    Supported:
    - Gaussian Splatting PLY
    - Standard PLY
    - GLB / GLTF / OBJ / STL
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File not found: {model_path}")

    ext = os.path.splitext(model_path)[1].lower()

    try:
        if ext == ".ply":
            ply = PlyData.read(model_path)
            props = ply["vertex"].data.dtype.names

            if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(props):
                _view_gaussian_ply(model_path)
            else:
                _view_standard_ply(model_path)

        elif ext in {".glb", ".gltf", ".obj", ".stl", ".fbx"}:
            _view_mesh(model_path)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except KeyboardInterrupt:
        print("\n[INFO] Viewer interrupted by user")
        sys.exit(0)
