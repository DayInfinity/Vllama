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
# Graceful Ctrl+C handler
# ------------------------------------------------------
def _handle_interrupt(sig, frame):
    print("\n[INFO] Viewer closed by user (Ctrl+C)")
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_interrupt)


# ------------------------------------------------------
# Pi3 PLY detection (ROBUST)
# ------------------------------------------------------
def _is_pi3_ply(pcd: o3d.geometry.PointCloud) -> bool:
    """
    Detect Pi3-generated PLY point clouds
    """
    if not pcd.has_colors():
        return False

    if len(pcd.points) < 50_000:
        return False

    # Missing or broken normals → Pi3
    if not pcd.has_normals():
        return True

    normals = np.asarray(pcd.normals)
    if normals.size == 0 or np.allclose(normals, 0):
        return True

    return False


# ------------------------------------------------------
# Gaussian Splatting PLY Viewer
# ------------------------------------------------------
def _view_gaussian_ply(ply_path: str):
    print("[INFO] Gaussian Splatting PLY detected")

    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    points = np.vstack([v["x"], v["y"], v["z"]]).T

    f_dc = np.vstack([
        v["f_dc_0"],
        v["f_dc_1"],
        v["f_dc_2"]
    ]).T

    colors = np.clip(expit(f_dc), 0.0, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


# ------------------------------------------------------
# Pi3 PLY Viewer
# ------------------------------------------------------
def _view_pi3_ply(ply_path: str):
    print("[INFO] Pi3 PLY detected")

    pcd = o3d.io.read_point_cloud(ply_path)

    # Normalize colors safely
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(colors, 0.0, 1.0)
        )

    # Remove broken normals
    pcd.normals = o3d.utility.Vector3dVector([])

    # Rotate 180° around X axis
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=pcd.get_center())

    vis = o3d.visualization.Visualizer()
    vis.create_window("Pi3 Viewer")
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = [0, 0, 0]
    opt.light_on = False

    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)

    vis.run()
    vis.destroy_window()


# ------------------------------------------------------
# Standard PLY Viewer
# ------------------------------------------------------
def _view_standard_ply(ply_path: str):
    print("[INFO] Standard PLY detected")

    pcd = o3d.io.read_point_cloud(ply_path)

    # Fix black models if normals exist but are broken
    if pcd.has_normals():
        pcd.normals = o3d.utility.Vector3dVector([])

    o3d.visualization.draw_geometries([pcd])


# ------------------------------------------------------
# Mesh Viewer
# ------------------------------------------------------
def _view_mesh(mesh_path: str):
    print("[INFO] Mesh format detected")

    tm_obj = trimesh.load(mesh_path, force="scene")

    if isinstance(tm_obj, trimesh.Scene):
        scene = pyrender.Scene.from_trimesh_scene(tm_obj)
    else:
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(tm_obj))

    pyrender.Viewer(scene, use_raymond_lighting=True)


# ------------------------------------------------------
# Unified Public API
# ------------------------------------------------------
def view_3d_model(model_path: str):
    """
    Automatically detect and display a 3D model.

    Supported:
    - Gaussian Splatting PLY
    - Pi3 PLY
    - Standard PLY
    - GLB / GLTF / OBJ / STL / FBX
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    ext = os.path.splitext(model_path)[1].lower()

    try:
        if ext == ".ply":
            ply = PlyData.read(model_path)
            props = ply["vertex"].data.dtype.names

            # Gaussian Splatting
            if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(props):
                _view_gaussian_ply(model_path)
                return

            # Load once for Pi3 / Standard detection
            pcd = o3d.io.read_point_cloud(model_path)

            if _is_pi3_ply(pcd):
                _view_pi3_ply(model_path)
            else:
                _view_standard_ply(model_path)

        elif ext in {".glb", ".gltf", ".obj", ".stl", ".fbx"}:
            _view_mesh(model_path)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except KeyboardInterrupt:
        print("\n[INFO] Viewer interrupted")
        sys.exit(0)
