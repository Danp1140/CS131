# Visualizes SfM data created using sfm.py
#
# Arg 1 is the location of the SfM model output (arg 3 of sfm.py)
#
# Creates plots showing
# - Points in 3D (interactive viewport through Open3D)
# - Depth map from the first camera's projection (in-progress)

import sys
import pycolmap as pcm
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

recon = pcm.Reconstruction(sys.argv[1])

points = np.array([p.xyz for p in recon.points3D.values()])
cols = np.array([p.color for p in recon.points3D.values()]).astype(np.float32) / 256

pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(points)
pc.colors = o3d.utility.Vector3dVector(cols)
# o3d.visualization.draw_geometries([pc])

pc = o3d.t.geometry.PointCloud.from_legacy(pc)

intr = recon.image(1).camera.calibration_matrix()
extr = recon.image(1).cam_from_world().matrix()
extr = np.vstack((extr, [0, 0, 0, 1]))
extr = np.linalg.inv(extr)

print(intr)
print(extr)

depth = pc.project_to_depth_image(
    800, 600, 
    intr, extrinsics=extr, 
    depth_scale=1, depth_max=1000000)

plt.figure()
plt.imshow(depth)

plt.show()

