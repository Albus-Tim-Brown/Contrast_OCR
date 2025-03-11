import numpy as np
import json
import cv2
import os
from shapely.geometry import Polygon

# TODO
seal_train_gt = "./seal_ppocr_gt/seal_det_img.txt"
seal_valid_gt = "./seal_ppocr_gt/seal_det_img.txt"

def gen_main_train_txt(mode='train'):
    if mode == "train":
        file_path = seal_train_gt
    if mode in ['valid', 'test']:
        file_path = seal_valid_gt

    save_path = f"./seal_VOC/ImageSets/Main/{mode}.txt"
    save_train_path = f"./seal_VOC/{mode}.txt"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    datas = open(file_path, 'r').readlines()
    img_names = []
    train_names = []
    for line in datas:
        img_name = line.strip().split('\t')[0]
        img_name = os.path.basename(img_name)
        (i_name, extension) = os.path.splitext(img_name)
        t_name = 'JPEGImages/'+str(img_name)+' '+'Annotations/'+str(i_name)+'.xml\n'
        train_names.append(t_name)
        img_names.append(i_name + "\n")

    with open(save_train_path, "w") as f:
        f.writelines(train_names)
        f.close()

    with open(save_path, "w") as f:
        f.writelines(img_names)
        f.close()

    print(f"{mode} save done")


def gen_xml_label(mode='train'):
    if mode == "train":
        file_path = seal_train_gt
    if mode in ['valid', 'test']:
        file_path = seal_valid_gt

    datas = open(file_path, 'r').readlines()
    img_names = []
    train_names = []
    anno_path = "./seal_VOC/Annotations"
    img_path = "./seal_VOC/JPEGImages"

    if not os.path.exists(anno_path):
        os.makedirs(anno_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for idx, line in enumerate(datas):
        img_name, label = line.strip().split('\t')
        img = cv2.imread(os.path.join("./seal_labeled_datas", img_name))
        cv2.imwrite(os.path.join(img_path, os.path.basename(img_name)), img)
        height, width, c = img.shape
        img_name = os.path.basename(img_name)
        (i_name, extension) = os.path.splitext(img_name)
        label = json.loads(label)

        xml_file = open(("./seal_VOC/Annotations" + '/' + i_name + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>seal_VOC</folder>\n')
        xml_file.write('    <filename>' + str(img_name) + '</filename>\n')
        xml_file.write('    <path>' + 'Annotations/' + str(img_name) + '</path>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')
        xml_file.write('    <segmented>0</segmented>\n')

        for anno in label:
            poly = anno['polys']
            if anno['cls'] == 1:
                gt_cls = 'redseal'
            xmin = np.min(np.array(poly)[:, 0])
            ymin = np.min(np.array(poly)[:, 1])
            xmax = np.max(np.array(poly)[:, 0])
            ymax = np.max(np.array(poly)[:, 1])
            xmin,ymin,xmax,ymax= int(xmin),int(ymin),int(xmax),int(ymax)
            xml_file.write('    <object>\n')
            xml_file.write('        <name>'+str(gt_cls)+'</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>'+str(xmin)+'</xmin>\n')
            xml_file.write('            <ymin>'+str(ymin)+'</ymin>\n')
            xml_file.write('            <xmax>'+str(xmax)+'</xmax>\n')
            xml_file.write('            <ymax>'+str(ymax)+'</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')
        xml_file.close()
    print(f'{mode} xml save done!')


gen_main_train_txt()
gen_main_train_txt('valid')
gen_xml_label('train')
gen_xml_label('valid')