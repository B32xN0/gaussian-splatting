import os

import numpy as np
from PIL import Image

from utils.graphics_utils import focal2fov


def extract_duster_caminfos(c2ws, imgs, focals, images_folder):
    cam_infos = []
    for i, (img_dust, c2w, focal) in enumerate(zip(imgs, c2ws, focals)):

        # extract R, T from inverted c2w
        # TODO convert cam parameters to larger size images
        # TODO maybe the parameters don't actually change ?
        # Colmap Saves World to Camera as Quaternion and T
        # They define R to be World to Camera from the Colmap Quaternion TRANSPOSED
        # Which should be equivalent to R from c2w (unitary part of matrix)?
        # But for simplicity sake lets just do it like that
        # Translation is supposed to be translation of w2c from colmap
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].transpose()
        T = w2c[:3, 3]

        # Compute Fov with width and height of the images Duster used to obtain focal length
        # From then on pixel width can be used from the original images, does not affect fov
        FovX = focal2fov(focal, img_dust.shape[1])
        FovY = focal2fov(focal, img_dust.shape[0])

        # open and extract actual images
        # TODO assumes that duster preserves order, hopefully it does
        # ENDING = ".jpeg"
        ENDING = ".png"
        INDEX_OFFSET = 1
        image_path = os.path.join(images_folder, "{:03d}".format(i + INDEX_OFFSET) + ENDING)
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        width, height = image.size

        from scene.dataset_readers import CameraInfo
        cam_infos.append(CameraInfo(uid=i, R=R, T=T, FovX=FovX, FovY=FovY,
                                    image=image, image_path=image_path, image_name=image_name,
                                    width=width, height=height))
    return cam_infos
