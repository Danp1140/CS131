# Generates a depth image from a COLMAP SfM reconstruction by reprojecting the points according to the first camera,
# gridding them, and looking for the lowest change in depth gradient (windowed average) starting from the bottom
#
# Arg 1 is SfM output folder
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

recon = pcm.Reconstruction(sys.argv[1])

pc = o3d.geometry.PointCloud()

if len(sys.argv) > 2 and Path(sys.argv[2]).is_file() and sys.argv[2][-4:] == ".ply":
    data = PlyData.read(sys.argv[2])
    pc.points = o3d.utility.Vector3dVector(np.array([[p["x"], p["y"], p["z"]] for p in data["vertex"]]))
    pc.colors = o3d.utility.Vector3dVector(np.array([[p["red"], p["green"], p["blue"]] for p in data["vertex"]]) / 256)

else:
    pc.points = o3d.utility.Vector3dVector(np.array([p.xyz for p in recon.points3D.values()]))
    pc.colors = o3d.utility.Vector3dVector(np.array([p.color for p in recon.points3D.values()]).astype(np.float32) / 256)

pc.estimate_normals()

intr = np.eye(4)
intr[:3, :3] = recon.image(1).camera.calibration_matrix()
extr = recon.image(1).cam_from_world().matrix()
extr = np.vstack((extr, [0, 0, 0, 1]))

# pc = pc.transform(np.linalg.inv(extr))
np.asarray(pc.points)[:, :2] = np.array([recon.image(1).project_point(p) for p in pc.points])

# cl, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# pc = pc.select_by_index(ind)

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
    N = 50
    k_w = 15
    out = np.zeros((n,N-2*k_w))
    out_dif = np.zeros((n, N-2*k_w-1))
    for i in range(n):
        trans_origins[i, :] = [min_x + (i+1) * del_x / (n+1), max_y]
        plt.plot(trans_origins[i, 0] * np.ones(2), [0, max_y])
        trans_points = []
        for p in pc.points:
            if abs(p[0] - trans_origins[i, 0]) < del_x / (n+1):
                trans_points += [p]
        trans_points = np.array(trans_points)
        trans_points = trans_points[np.argsort(trans_points[:, 1]), :]
        #plt.figure()
        #plt.plot(trans_points[:, 2])
        # out[i, :] = np.interp(np.linspace(np.min(trans_points[:, 1]), np.max(trans_points[:, 1]), N), trans_points[:, 1], trans_points[:, 2])
        temp = np.interp(np.linspace(0, len(trans_points[:, 1]), N), np.arange(0, len(trans_points[:, 1])), trans_points[:, 2])
        out[i, :] = np.convolve(temp, np.ones(2*k_w+1), mode="valid")
        out_dif[i, :] = np.diff(out[i, :])
        out_dif[i, :] = np.abs(out_dif[i, :])
        # min_points[i] = trans_points[sp.signal.argrelextrema(trans_points[:, 2], np.greater)[0][-1], 2]
        min_points[i] = min_y + (max_y - min_y) * (1 - sp.signal.argrelextrema(out_dif[i, :], np.greater)[0][0] / len(out_dif[i, :]))
    plt.plot(trans_origins[:, 0], min_points)
    print(min_points)
    return out, out_dif
 

# img = cv.imread("imgs/miramar_2013/miramar_1.png")
img = cv.imread("imgs/carlsbad_2024/carlsbad_1.png")

plt.figure()

plt.imshow(img[:, :, ::-1])
gm = img.shape[0] * gm / 500
plt.plot([0, img.shape[1]], [gm, gm])
plt.scatter(np.asarray(pc.points)[:, 0], np.asarray(pc.points)[:, 1], c=np.asarray(pc.colors))
#trans, dtrans = genTransects(10)

# for i in range(10):
    # plt.figure()
    # plt.plot(dtrans[i, :])

plt.show()

