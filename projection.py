"""projects 3D points onto image"""

import os
from pprint import pprint
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pykitti

matplotlib.use('Agg')

# read the arguments
ARGS_FILE = 'projection-args.json'

with open(ARGS_FILE, 'r', encoding='utf8') as f:
    args = json.load(f)

outputdir         : str = args['output path']
basedir           : str = args['dataset path']
calib_cam_to_cam  : str = args['cam to cam']
calib_velo_to_cam : str = args['velo to cam']
# velo              : str = args['velo']
date              : str = args['date']
drive             : str = args['drive']
cam_no            : str = args['camera no']
img_no            : int = args['image no']

# create the output directory
os.makedirs(outputdir, exist_ok=True)

# load the dataset
dataset = pykitti.raw(basedir, date, drive) # type: ignore

# load the image data
if cam_no == "00":
    img = np.asarray(dataset.get_cam0(img_no))
elif cam_no == "01":
    img = np.asarray(dataset.get_cam1(img_no))
elif cam_no == "02":
    img = np.asarray(dataset.get_cam2(img_no))
elif cam_no == "03":
    img = np.asarray(dataset.get_cam3(img_no))

# save the image
pprint(f'Image shape: {img.shape}')
plt.imshow(img)
plt.savefig(os.path.join(outputdir, f'img_{img_no:06}.png'))
