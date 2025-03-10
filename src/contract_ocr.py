import os

import fitz
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
import logging

logging.basicConfig(level=logging.DEBUG)


def extract_text_from_pdf(pdf_path, ocr, page_num):
    result = ocr.ocr(pdf_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        if res is None:
            logging.debug(f"Empty page {idx + 1} detected, skip it.")
            continue
        for line in res:
            print(line)
    return result


def save_ocr_results(result, imgs, output_dir):
    for idx in range(len(result)):
        res = result[idx]
        if res is None:
            continue
        image = imgs[idx]
        boxes = [line[0] for line in res]
        txts = [line[1][0] for line in res]
        scores = [line[1][1] for line in res]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save(f'{output_dir}/result_page_{idx}.jpg')


def process_pdf_ocr(pdf_path, output_dir, page_num=4):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=page_num, use_gpu=0)
    result = extract_text_from_pdf(pdf_path, ocr, page_num)

    imgs = []
    with fitz.open(pdf_path) as pdf:
        for pg in range(0, page_num):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)

    save_ocr_results(result, imgs, output_dir)


# TODO Seal OCR
def process_pdf_seal_ocr(pdf_path, output_dir):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch',
                    det_db_unclip_ratio=2.5,
                    max_side_len=2000)

    with fitz.open(pdf_path) as pdf:
        for pg in range(len(pdf)):
            page = pdf[pg]
            pix = page.getPixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            img = cv2.cvtColor(np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples)),
                               cv2.COLOR_RGB2BGR)

            # seal_boxes = detect_seal_area(img)
            seal_boxes = None
            for i, box in enumerate(seal_boxes):
                x1, y1, x2, y2 = box
                seal_img = img[y1:y2, x1:x2]

                seal_result = ocr.ocr(seal_img, cls=True)
                for line in seal_result:
                    print(f"Page{pg + 1} Seal{i + 1}: {line[1][0]}")

                # cv2.imwrite(f"{output_dir}/seal_{pg}_{i}.jpg", seal_img)


if __name__ == '__main__':
    PAGE_NUM = 4

    # Local Files
    input_dir = "../input/pdf/"
    output_dir = '../output/contract/'
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            output_dir = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Processing: {pdf_path}")
            process_pdf_ocr(pdf_path, output_dir, page_num=PAGE_NUM)

    # TODO Network Files
