import json
import os

import cv2
import numpy as np

# TODO
def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def run(data_dir, label_file, save_dir):
    datas = open(label_file, 'r').readlines()
    for idx, line in enumerate(datas):
        img_path, label = line.strip().split('\t')
        img_path = os.path.join(data_dir, img_path)

        label = json.loads(label)
        src_im = cv2.imread(img_path)
        if src_im is None:
            continue

        # TODO text_boxes
        text_boxes = []
        for anno in label:
            seal_box = anno['seal_box']
            txt_boxes = anno['polys']
            crop_im = get_rotate_crop_image(src_im, text_boxes)

            save_path = os.path.join(save_dir, f'{idx}.png')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # print(src_im.shape)
            cv2.imwrite(save_path, crop_im)