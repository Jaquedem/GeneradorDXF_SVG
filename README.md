# Image to CAD/CAM Converter 🛠️

A Python-based tool designed to convert raster images (JPEG/PNG) of 3D printed objects or sketches into engineering-grade vector formats (DXF/SVG) and 3D models (STL).

Designed for reverse engineering, CNC cutting, and laser engraving workflows.

## 🚀 Features

- **High Precision Tracing:** Uses advanced Computer Vision (OpenCV) to detect contours with sub-pixel accuracy.
- **Noise Filtering:** Implements hierarchy-based logic to eliminate "double lines" caused by lighting artifacts on 3D objects.
- **DXF Export:** Generates clean `LWPOLYLINE` entities compatible with AutoCAD, Fusion 360, and CNC software.
- **STL Generation:** Extrudes 2D contours into 3D meshes using `trimesh` and `shapely`.
- **Interactive Editor:** Includes a visual selector (Matplotlib) to manually toggle active contours before exporting to STL.

## 📋 Prerequisites

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
