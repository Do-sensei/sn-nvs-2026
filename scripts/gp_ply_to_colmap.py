#!/usr/bin/env python3
"""Convert a 3DGS Gaussian-splat point_cloud.ply into a COLMAP-style
RGB point cloud (x, y, z + zero normal + uchar red/green/blue).

Step 1 (GaussianPro) produces a Gaussian-splat ply (f_dc / opacity /
scale / rotation per point). Step 2 (3DGS) reads its init cloud through
`fetchPly`, which requires plain `red/green/blue` vertex properties.
This script keeps only the Gaussian centres and their SH degree-0
colour; opacity / scale / rotation / higher-order SH are dropped.

Usage: gp_ply_to_colmap.py <gaussian_splat.ply> <out_points3D.ply>
"""
import sys
import numpy as np
from plyfile import PlyData, PlyElement

# SH degree-0 basis function value (1 / (2*sqrt(pi))); matches the 3DGS
# SH2RGB convention used by gaussian-splatting/utils/sh_utils.py.
C0 = 0.28209479177387814


def convert(src, dst):
    ply = PlyData.read(src)
    v = ply["vertex"]
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    f_dc = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float32)
    # Match the original one-off conversion exactly: clip to [0,1] then
    # truncate (np.uint8 cast floors; no rounding).
    rgb = np.clip(f_dc * C0 + 0.5, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    normals = np.zeros_like(xyz)

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"], elements["y"], elements["z"] = xyz.T
    elements["nx"], elements["ny"], elements["nz"] = normals.T
    elements["red"], elements["green"], elements["blue"] = rgb.T
    PlyData([PlyElement.describe(elements, "vertex")]).write(dst)
    print(f"[convert] {xyz.shape[0]} pts  {src} -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
