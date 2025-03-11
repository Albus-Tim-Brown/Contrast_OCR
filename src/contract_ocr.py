import os

import fitz
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
import logging

logging.basicConfig(level=logging.DEBUG)


def extract_text_from_pdf(pdf_path, ocr, page_num, threshold=0.8):
    result = ocr.ocr(pdf_path, cls=True)
    processed_results = []
    for page_idx, page_result in enumerate(result):
        if page_result is None:
            logging.debug(f"Empty page {page_idx + 1} detected, skip it.")
            continue
        for detection in page_result:
            box, (text, score) = detection
            if score < threshold:
                continue
            flat_box = [coordinate for point in box for coordinate in point]
            result_dict = {'box': flat_box, 'text': text}
            print(result_dict)
            processed_results.append(result_dict)
    return processed_results


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
    processed_results = extract_text_from_pdf(pdf_path, ocr, page_num, threshold=0.8)

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

    save_ocr_results(result=ocr.ocr(pdf_path, cls=True), imgs=imgs, output_dir=output_dir)
    return processed_results

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
