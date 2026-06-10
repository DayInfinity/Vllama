# `vllama view3d` — 3D Model Viewer

Open and interactively view 3D model files in a local viewer window.

---

## Syntax

```bash
vllama view3d --path <model_file>
```

---

## Supported Formats

| Format | Extension |
|---|---|
| Polygon File Format | `.ply` |
| GL Transmission Format | `.glb` / `.gltf` |
| Wavefront OBJ | `.obj` |
| Stereolithography | `.stl` |
| Filmbox | `.fbx` |

---

## Examples

```bash
vllama view3d --path model.ply
vllama view3d --path scan.glb
vllama view3d --path object.obj
```

---

## Common Workflow

```bash
# Step 1: Generate a 3D model from an image on Kaggle GPU
vllama image3d --path photo.jpg --service kaggle -o ./3d_outputs

# Step 2: View the result
vllama view3d --path ./3d_outputs/output.ply
```

The viewer opens in a native window with mouse controls for rotating, zooming, and panning the model.
