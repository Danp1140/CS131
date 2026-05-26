# Plots a disparity map between two images
#
# Arg 1 is the left image filepath
# Arg 2 is the right
#
# Produces several plots:
# - Distribution of best to second-best descriptor distance to evaluate SIFT match quality
# - Side-by-side rectified images to evaluate rectification quality
# - Disparity map; hoping to see a gradient front-to-back with an area of little change between (backshore)
# 
# Default parameters/operations that may be tweaked:
# - Converted to grayscale
# - Downscaled 0.1x
# - Sift best to second-best ratio cutoff 0.75
# - RANSAC threshold 5


import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def prepImg(path, scale=0.1):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    return img

def rectify(l, r, sift_ratio=0.75, ransac_ratio=5):
    sift = cv.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(l, None)
    kp2, desc2 = sift.detectAndCompute(r, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    ratios = []
    good_ratios = []
    good_matches = []
    for m, n in matches:
        ratios.append(m.distance / n.distance)
        if ratios[-1] < sift_ratio:
            good_matches.append(m)
            good_ratios.append(ratios[-1])
    ratios = np.array(ratios)
    good_ratios = np.array(good_ratios)
    good_matches = np.array(good_matches)

    epdf = sp.stats.gaussian_kde(ratios)
    x = np.linspace(0, 1, 100)
    plt.figure()
    plt.plot(x, epdf.pdf(x))
    plt.xlabel("Best to Second-Best Distance Ratio")
    plt.ylabel("Frequency")

    good_matches = good_matches[good_ratios.argsort()[::-1]]

    p1 = np.array([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in good_matches])
    p2 = np.array([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in good_matches])
    F, mask = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, ransac_ratio)
    success, H1, H2 = cv.stereoRectifyUncalibrated(p1, p2, F, l.shape[::-1], threshold=ransac_ratio)
    return H1, H2

def disparity(l, r, minDisp=32, numDisp=512, bSize=63):
    stereo = cv.StereoSGBM_create(minDisparity=minDisp, numDisparities=numDisp, blockSize=bSize)
    return stereo.compute(l, r).astype(np.float32)

img_l = prepImg(sys.argv[1]) 
img_r = prepImg(sys.argv[2]) 

H_l, H_r = rectify(img_l, img_r)

img_l = cv.warpPerspective(img_l, H_l, img_l.shape[::-1])
img_r = cv.warpPerspective(img_r, H_r, img_r.shape[::-1])

disp = disparity(img_l, img_r)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_l)
plt.subplot(1, 2, 2)
plt.imshow(img_r)

plt.figure()
plt.imshow(disp, cmap="gray")

plt.show()
