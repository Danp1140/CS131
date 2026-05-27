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

recon = pcm.Reconstruction(sys.argv[1])

points = np.array([p.xyz for p in recon.points3D.values()])
cols = np.array([p.color for p in recon.points3D.values()]).astype(np.float32) / 256

pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(points)
pc.colors = o3d.utility.Vector3dVector(cols)
pc.estimate_normals()

intr = recon.image(1).camera.calibration_matrix()
extr = recon.image(1).cam_from_world().matrix()
extr = np.vstack((extr, [0, 0, 0, 1]))

pc = pc.transform(np.linalg.inv(extr))

print(intr)
print(extr)

print(f"center: {pc.get_center()}")

points = np.array(pc.points)

grid_x, grid_y = np.meshgrid(np.linspace(min(points[:, 0]), max(points[:, 0]), 500), np.linspace(min(points[:, 1]), max(points[:, 1]), 500))
grid_z = sp.interpolate.griddata((points[:, 0], points[:, 1]), points[:, 2], (grid_x, grid_y), method='cubic')

plt.figure()
pcol = plt.pcolormesh(grid_x, -grid_y, grid_z, shading='auto', cmap='gray').get_array()

plt.figure()
elev = np.nanmean(pcol, axis=1)
plt.plot(elev)

plt.figure()
grad = np.convolve(np.diff(elev), np.ones(15), mode="valid")
plt.plot(grad)

plt.figure()
sob = np.array(cv.Sobel(np.array(pcol), cv.CV_64F, 0, 1, ksize=31))
gm = sp.signal.argrelextrema(grad, np.greater)[0][-1]
print(sob.shape)
print(f"grad min idx {gm}")
plt.imshow(sob)
plt.plot([0, sob.shape[1]], [gm, gm])

plt.show()

