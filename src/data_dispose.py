import numpy as np
import json
import cv2
import os

from matplotlib.bezier import get_intersection
from paddleocr.ppocr.data.imaug.copy_paste import get_union
from shapely.geometry import Polygon


def poly2box(poly):
    xmin = np.min(np.array(poly)[:, 0])
    ymin = np.min(np.array(poly)[:, 1])
    xmax = np.max(np.array(poly)[:, 0])
    ymax = np.max(np.array(poly)[:, 1])
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def draw_text_det_res(dt_boxes, src_im, color=(255, 255, 0)):
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=color, thickness=2)
    return src_im

class LabelDecode(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        label = json.loads(data['label'])

        nBox = len(label)
        seal_boxes = self.get_seal_boxes(label)

        gt_label = []

        for seal_box in seal_boxes:
            seal_anno = {'seal_box': seal_box}
            boxes, txts, txt_tags = [], [], []

            for bno in range(0, nBox):
                box = label[bno]['points']
                txt = label[bno]['transcription']
                try:
                    ints = self.get_intersection(box, seal_box)
                except Exception as E:
                    print(E)
                    continue

                if (abs(Polygon(box).area - self.get_intersection(box, seal_box))
                        < 1e-3 < abs(Polygon(box).area - self.get_union(box, seal_box))):

                    boxes.append(box)
                    txts.append(txt)
                    if txt in ['*', '###', '待识别']:
                        txt_tags.append(True)
                    else:
                        txt_tags.append(False)

            seal_anno['polys'] = boxes
            seal_anno['texts'] = txts
            seal_anno['ignore_tags'] = txt_tags

            gt_label.append(seal_anno)

        return gt_label

    def get_seal_boxes(self, label):

        nBox = len(label)
        seal_box = []
        for bno in range(0, nBox):
            box = label[bno]['points']
            if len(box) == 4:
                seal_box.append(box)

        if len(seal_box) == 0:
            return None

        seal_box = self.valid_seal_box(seal_box)
        return seal_box


    def is_seal_box(self, box, boxes):
        is_seal = True
        for poly in boxes:
            if list(box.shape()) != list(box.shape.shape()):
                if abs(Polygon(box).area - self.get_intersection(box, poly)) < 1e-3:
                    return False
            else:
                if np.sum(np.array(box) - np.array(poly)) < 1e-3:
                    # continue when the box is same with poly
                    continue
                if abs(Polygon(box).area - self.get_intersection(box, poly)) < 1e-3:
                    return False
        return is_seal


    def valid_seal_box(self, boxes):
        if len(boxes) == 1:
            return boxes

        new_boxes = []
        flag = True
        for k in range(0, len(boxes)):
            flag = True
            tmp_box = boxes[k]
            for i in range(0, len(boxes)):
                if k == i: continue
                if abs(Polygon(tmp_box).area - self.get_intersection(tmp_box, boxes[i])) < 1e-3:
                    flag = False
                    continue
            if flag:
                new_boxes.append(tmp_box)

        return new_boxes


    def get_union(self, pD, pG):
        return Polygon(pD).union(Polygon(pG)).area

    def get_intersection_over_union(self, pD, pG):
        return get_intersection(pD, pG) / get_union(pD, pG)

    def get_intersection(self, pD, pG):
        return Polygon(pD).intersection(Polygon(pG)).area

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes


def gen_extract_label(data_dir, label_file, seal_gt, seal_ppocr_gt):
    label_decode_func = LabelDecode()
    gts = open(label_file, "r").readlines()

    seal_gt_list = []
    seal_ppocr_list = []

    for idx, line in enumerate(gts):
        img_path, label = line.strip().split("\t")
        data = {'label': label, 'img_path':img_path}
        res = label_decode_func(data)
        src_img = cv2.imread(os.path.join(data_dir, img_path))
        if res is None:
            print("ERROR! res is None!")
            continue

        anno = []
        for i, gt in enumerate(res):
            # print(i, box, type(box), )
            anno.append({'polys': gt['seal_box'], 'cls':1})

        seal_gt_list.append(f"{img_path}\t{json.dumps(anno)}\n")
        seal_ppocr_list.append(f"{img_path}\t{json.dumps(res)}\n")

    if not os.path.exists(os.path.dirname(seal_gt)):
        os.makedirs(os.path.dirname(seal_gt))
    if not os.path.exists(os.path.dirname(seal_ppocr_gt)):
        os.makedirs(os.path.dirname(seal_ppocr_gt))

    with open(seal_gt, "w") as f:
        f.writelines(seal_gt_list)
        f.close()

    with open(seal_ppocr_gt, 'w') as f:
        f.writelines(seal_ppocr_list)
        f.close()

def vis_seal_ppocr(data_dir, label_file, save_dir):

    datas = open(label_file, 'r').readlines()
    for idx, line in enumerate(datas):
        img_path, label = line.strip().split('\t')
        img_path = os.path.join(data_dir, img_path)

        label = json.loads(label)
        src_im = cv2.imread(img_path)
        if src_im is None:
            continue

        for anno in label:
            seal_box = anno['seal_box']
            txt_boxes = anno['polys']

             # vis seal box
            src_im = draw_text_det_res([seal_box], src_im, color=(255, 255, 0))
            src_im = draw_text_det_res(txt_boxes, src_im, color=(255, 0, 0))

        save_path = os.path.join(save_dir, os.path.basename(img_path))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # print(src_im.shape)
        cv2.imwrite(save_path, src_im)


def draw_html(img_dir, save_name):
    import glob

    images_dir = glob.glob(img_dir + "/*")
    print(len(images_dir))

    html_path = save_name
    with open(html_path, 'w') as html:
        html.write('<html>\n<body>\n')
        html.write('<table border="1">\n')
        html.write("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />")

        html.write("<tr>\n")
        html.write(f'<td> \n GT')

        for i, filename in enumerate(sorted(images_dir)):
            if filename.endswith("txt"): continue
            print(filename)

            base = "{}".format(filename)
            if True:
                html.write("<tr>\n")
                html.write(f'<td> {filename}\n GT')
                html.write('<td>GT 310\n<img src="%s" width=640></td>' % (base))
                html.write("</tr>\n")

        html.write('<style>\n')
        html.write('span {\n')
        html.write('    color: red;\n')
        html.write('}\n')
        html.write('</style>\n')
        html.write('</table>\n')
        html.write('</html>\n</body>\n')
    print("ok")


def crop_seal_from_img(label_file, data_dir, save_dir, save_gt_path):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datas = open(label_file, 'r').readlines()
    all_gts = []
    count = 0
    for idx, line in enumerate(datas):
        img_path, label = line.strip().split('\t')
        img_path = os.path.join(data_dir, img_path)

        label = json.loads(label)
        src_im = cv2.imread(img_path)
        if src_im is None:
            continue

        for c, anno in enumerate(label):
            seal_poly = anno['seal_box']
            txt_boxes = anno['polys']
            txts = anno['texts']
            ignore_tags = anno['ignore_tags']

            box = poly2box(seal_poly)
            img_crop = src_im[box[0][1]:box[2][1], box[0][0]:box[2][0], :]

            save_path = os.path.join(save_dir, f"{idx}_{c}.jpg")
            cv2.imwrite(save_path, np.array(img_crop))

            img_gt = []
            for i in range(len(txts)):
                txt_boxes_crop = np.array(txt_boxes[i])
                txt_boxes_crop[:, 1] -= box[0, 1]
                txt_boxes_crop[:, 0] -= box[0, 0]
                img_gt.append({'transcription': txts[i], "points": txt_boxes_crop.tolist(), "ignore_tag": ignore_tags[i]})

            if len(img_gt) >= 1:
                count += 1
            save_gt = f"{os.path.basename(save_path)}\t{json.dumps(img_gt)}\n"

            all_gts.append(save_gt)

    print(f"The num of all image: {len(all_gts)}, and the number of useful image: {count}")
    if not os.path.exists(os.path.dirname(save_gt_path)):
        os.makedirs(os.path.dirname(save_gt_path))

    with open(save_gt_path, "w") as f:
        f.writelines(all_gts)
        f.close()
    print("Done")


if __name__ == "__main__":
    # 数据处理
    gen_extract_label("./seal_labeled_datas",
                      "./seal_labeled_datas/Label.txt",
                      "./seal_ppocr_gt/seal_det_img.txt",
                      "./seal_ppocr_gt/seal_ppocr_img.txt")

    vis_seal_ppocr("./seal_labeled_datas",
                   "./seal_ppocr_gt/seal_ppocr_img.txt",
                   "./seal_ppocr_gt/seal_ppocr_vis/")

    draw_html("./seal_ppocr_gt/seal_ppocr_vis/", "./vis_seal_ppocr.html")

    seal_ppocr_img_label = "./seal_ppocr_gt/seal_ppocr_img.txt"

    crop_seal_from_img(seal_ppocr_img_label,
                       "./seal_labeled_datas/",
                       "./seal_img_crop",
                       "./seal_img_crop/label.txt")