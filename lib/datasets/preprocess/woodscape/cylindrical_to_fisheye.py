import os
import numpy as np
import cv2, os, glob
from scipy.spatial.transform import Rotation as SciRot
from projection import Camera, RadialPolyCamProjection, CylindricalProjection, read_cam_from_json, \
    create_img_projection_maps

woodscape_dir = '/raid/Research/dataset/tfzhou/woodscape/cylindrical'
save_dir = os.path.join(woodscape_dir, 'cylindrical_to_fisheye')
os.makedirs(save_dir, exist_ok=True)

sets = ['test']

for set in sets:
    os.makedirs(os.path.join(save_dir, set), exist_ok=True)


def make_cylindrical_cam(cam: Camera):
    """generates a cylindrical camera with a centered horizon"""
    assert isinstance(cam.lens, RadialPolyCamProjection)
    # creates a cylindrical projection
    lens = CylindricalProjection(cam.lens.coefficients[0])
    rot_zxz = SciRot.from_matrix(cam.rotation).as_euler('zxz')
    # adjust all angles to multiples of 90 degree
    rot_zxz = np.round(rot_zxz / (np.pi / 2)) * (np.pi / 2)
    # center horizon
    rot_zxz[1] = np.pi / 2
    # noinspection PyArgumentList
    return Camera(
        rotation=SciRot.from_euler(angles=rot_zxz, seq='zxz').as_matrix(),
        translation=cam.translation,
        lens=lens,
        size=cam.size, principle_point=(cam.cx_offset, cam.cy_offset),
        aspect_ratio=cam.aspect_ratio
    )


for set in sets:
    imagefiles = glob.glob(os.path.join(woodscape_dir, set, '*.png'))
    for imagefile in imagefiles:
        print(set, imagefile)
        basename = os.path.basename(imagefile)
        splits = basename.split('_')
        type = splits[-1][:-4]

        fisheye_cam = read_cam_from_json('{}.json'.format(type))
        cylindrical_cam = make_cylindrical_cam(fisheye_cam)

        cylindrical_image = cv2.imread(imagefile)

        map1, map2 = create_img_projection_maps(cylindrical_cam, fisheye_cam)
        fisheye_image = cv2.remap(cylindrical_image, map1, map2, cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(save_dir, set, basename), fisheye_image)


