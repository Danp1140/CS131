# Generates a depth image from a COLMAP SfM reconstruction by reprojecting the points according to the first camera,
# gridding them, and looking for the lowest change in depth gradient (windowed average) starting from the bottom
#
# Arg 1 is SfM output folder
# Arg 2 is the image name prefix (e.g., imgs/carlsbad_2024/carlsbad_)
# Arg 3, if provided, is a PLY point cloud for a dense reconstruction
#
# Makes plots of
# - Raw projected z coord gridded
# - Raw projected average of z coord across image
# - Windowed average of gradient of average of z coord across image
# - Sobel of gridded z coord with found backshore as a horizontal line

import sys
import pycolmap as pcm
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2 as cv
from pathlib import Path
from plyfile import PlyData

img_idx = 4

recon = pcm.Reconstruction(sys.argv[1])

pc = o3d.geometry.PointCloud()

if len(sys.argv) > 3 and Path(sys.argv[3]).is_file() and sys.argv[3][-4:] == ".ply":
    data = PlyData.read(sys.argv[3])
    pc.points = o3d.utility.Vector3dVector(np.array([[p["x"], p["y"], p["z"]] for p in data["vertex"]]))
    pc.colors = o3d.utility.Vector3dVector(np.array([[p["red"], p["green"], p["blue"]] for p in data["vertex"]]) / 256)

else:
    pc.points = o3d.utility.Vector3dVector(np.array([p.xyz for p in recon.points3D.values()]))
    pc.colors = o3d.utility.Vector3dVector(np.array([p.color for p in recon.points3D.values()]).astype(np.float32) / 256)

pc.estimate_normals()

intr = np.eye(4)
intr[:3, :3] = recon.image(img_idx).camera.calibration_matrix()
extr = recon.image(img_idx).cam_from_world().matrix()
extr = np.vstack((extr, [0, 0, 0, 1]))

# pc = pc.transform(np.linalg.inv(extr))
np.asarray(pc.points)[:, :2] = np.array([recon.image(img_idx).project_point(p) for p in pc.points])

cl, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pc = pc.select_by_index(ind)

points = np.array(pc.points)

grid_x, grid_y = np.meshgrid(np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 500), np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 500))
grid_z = sp.interpolate.griddata((points[:, 0], points[:, 1]), points[:, 2], (grid_x, grid_y), method='cubic')

plt.figure()
pcol = plt.pcolormesh(grid_x, grid_y, grid_z).get_array()

plt.figure()
plt.scatter(points[:, 0], points[:, 1], c=np.asarray(pc.colors)/256)

plt.figure()
elev = np.nanmean(pcol, axis=1)
grad = np.convolve(np.diff(elev), np.ones(31), mode="valid")
plt.plot(grad)

plt.figure()
sob = np.array(cv.Sobel(np.array(pcol), cv.CV_64F, 0, 1, ksize=31))
gm = sp.signal.argrelextrema(grad, np.greater)[0][-1]
print(f"grad min idx {gm}")
plt.imshow(sob)
plt.plot([0, sob.shape[1]], [gm, gm])

def genTransects(n):
    trans_norm = np.array([0, -1])
    trans_origins = np.zeros((n, 2)) 
    min_x = np.min(np.asarray(pc.points)[:, 0])
    max_x = np.max(np.asarray(pc.points)[:, 0])
    del_x = max_x - min_x
    min_y = np.min(np.asarray(pc.points)[:, 1])
    max_y = np.max(np.asarray(pc.points)[:, 1])
    min_points = np.zeros(n)
    N = 128
    k_w = 5
    out = np.zeros((n,N-2*k_w))
    out_dif = np.zeros((n, N-4*k_w-1))
    for i in range(n):
        trans_origins[i, :] = [min_x + (i+1) * del_x / (n+1), max_y]
        plt.plot(trans_origins[i, 0] * np.ones(2), [0, max_y], ":", color="k")
        trans_points = []
        for p in pc.points:
            if abs(p[0] - trans_origins[i, 0]) < del_x / (n+1) / 2:
                trans_points += [p]
        trans_points = np.array(trans_points)
        trans_points = trans_points[np.argsort(trans_points[:, 1]), :]
        min_y_trans = np.min(trans_points[:, 1])
        max_y_trans = np.max(trans_points[:, 1])
        temp = sp.interpolate.interp1d(trans_points[:, 1], trans_points[:, 2])(np.linspace(min_y_trans, max_y_trans, N))
        out[i, :] = np.convolve(temp, np.ones(2*k_w+1)/(2*k_w+1), mode="valid")
        out_dif[i, :] = np.convolve(np.diff(out[i, :]), np.ones(2*k_w+1) / (2*k_w+1), mode="valid")
        out_dif[i, :] = np.abs(out_dif[i, :])
        min_idx = sp.signal.argrelextrema(out_dif[i, :], np.greater)
        if len(min_idx[0]) > 0:
            min_points[i] = min_y_trans + (max_y_trans - min_y_trans) * float(min_idx[0][-1]) / float(len(out_dif[i, :]))
    plt.plot(trans_origins[:, 0], min_points, c="red")
    print(min_points)
    return out, out_dif
 

img = cv.imread(f"{argv[2]}{int(img_idx)}.png")

plt.figure()

plt.imshow(img[:, :, ::-1])
gm = img.shape[0] * gm / 500
n=16
trans, dtrans = genTransects(n)

# below may be uncommented to see each individual transect plotted
"""
for i in range(n):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(trans[i, :])
    plt.title("Transect z")
    plt.subplot(1, 2, 2)
    plt.plot(dtrans[i, :])
    plt.title("Transect dz")
"""

plt.show()

