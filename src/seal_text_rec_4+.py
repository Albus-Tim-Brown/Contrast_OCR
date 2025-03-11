import cv2
import numpy as np
import os
import json

# TODO
def get_polygon_crop_image(img, points):
    points = np.array(points, dtype=np.int32)
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    result = cv2.bitwise_and(img, mask)
    x, y, w, h = cv2.boundingRect(points)
    cropped_result = result[y:y+h, x:x+w]
    return cropped_result


def run(data_dir, label_file, save_dir, output_txt):
    if not os.path.exists(output_txt):
        os.makedirs(os.path.dirname(output_txt), exist_ok=True)
        open(output_txt, 'w').close()

    with open(output_txt, 'w') as txt_file:
        datas = open(label_file, 'r').readlines()
        for line in datas:
            filename, label = line.strip().split('\t')
            img_path = os.path.join(data_dir, filename)
            label = json.loads(label)
            src_im = cv2.imread(img_path)
            if src_im is None:
                continue
            for i, anno in enumerate(label):
                txt_boxes = anno['points']
                crop_im = get_polygon_crop_image(src_im, txt_boxes)
                crop_img_name = f'{filename.split("/")[-1].split(".")[0]}_crop_{i}.jpg'
                save_path = os.path.join(save_dir, crop_img_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(save_path, crop_im)
                txt_file.write(f'crop_img/{crop_img_name}\t{anno["transcription"]}\n')

if __name__ == "__main__":
    data_dir = "seal" # 图片数据集路径
    label_file = "Label.txt" # 数据标记结果txt路径
    save_dir = "crop_img" #导出识别结果，即识别所用训练图片的文件夹路径
    output_txt = "rec_gt.txt" # 识别训练标记结果txt路径
    run(data_dir, label_file, save_dir, output_txt)