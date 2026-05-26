# Constructs a point cloud from multiple images using COLMAP's structure-from-motion (SfM) algorithm)
# Uses incremental_mapping (as opposed to stereo_fusion)
#
# Arg 1 is the path of a folder containing ONLY the images to process
# Arg 2 is the path of the desired database location (should end in .db)
# Arg 3 is the path of an existing folder in which model outputs should be put
#
# Outputs a sparse reconstruction at the given path which can be read using visualize_sfm.py which in turn reads it

import sys
from pathlib import Path
import pycolmap as pcm

imgs = Path(sys.argv[1])
db = Path(sys.argv[2])
out = Path(sys.argv[3])

pcm.extract_features(db, imgs)
pcm.match_exhaustive(db)
recon = pcm.incremental_mapping(db, imgs, out)
recon[0].write(out)

