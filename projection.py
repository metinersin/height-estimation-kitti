"""projects 3D points onto image"""

import os
from pprint import pprint
import pickle as pkl
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

# read the image
if cam_no == "00":
    img = np.asarray(dataset.get_cam0(img_no))
elif cam_no == "01":
    img = np.asarray(dataset.get_cam1(img_no))
elif cam_no == "02":
    img = np.asarray(dataset.get_cam2(img_no))
elif cam_no == "03":
    img = np.asarray(dataset.get_cam3(img_no))
else:
    raise ValueError(f'Invalid camera number: {cam_no}')

img_height = img.shape[0]
img_width = img.shape[1]

# save the image
pprint(f'Image shape: {img.shape}')
plt.imshow(img)
plt.savefig(os.path.join(outputdir, f'img_{img_no:06}.png'))

# read the velodyne points
points_velo = dataset.get_velo(img_no)
pprint(f'Velodyne points shape: {points_velo.shape}')

# save the velodyne points
with open(os.path.join(outputdir, f'velo_{img_no:06}.pkl'), 'wb') as f:
    pkl.dump(points_velo, f)

# read the calibration matrices
if cam_no == "00":
    T_cam_velo : np.ndarray = dataset.calib.T_cam0_velo
    Rrect      : np.ndarray = dataset.calib.R_rect_00
    Prect      : np.ndarray = dataset.calib.P_rect_00
elif cam_no == "01":
    T_cam_velo = dataset.calib.T_cam1_velo
    Rrect      = dataset.calib.R_rect_01
    Prect      = dataset.calib.P_rect_01
elif cam_no == "02":
    T_cam_velo = dataset.calib.T_cam2_velo
    Rrect      = dataset.calib.R_rect_02
    Prect      = dataset.calib.P_rect_02
elif cam_no == "03":
    T_cam_velo = dataset.calib.T_cam3_velo
    Rrect      = dataset.calib.R_rect_03
    Prect      = dataset.calib.P_rect_03
else:
    raise ValueError(f'Invalid camera number: {cam_no}')

pprint(f'T_cam_velo: {T_cam_velo}')
pprint(f'Rrect: {Rrect}')
pprint(f'Prect: {Prect}')

# save the calibration matrices
np.savetxt(os.path.join(outputdir, f'T_cam_velo_{img_no:06}.txt'), T_cam_velo)
np.savetxt(os.path.join(outputdir, f'Rrect_{img_no:06}.txt'), Rrect)
np.savetxt(os.path.join(outputdir, f'Prect_{img_no:06}.txt'), Prect)
