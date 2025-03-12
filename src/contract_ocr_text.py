"""
使用PaddleOCR识别pdf合同文件，不识别最后一页
"""
import os
import re

import fitz
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
import logging

logging.basicConfig(level=logging.DEBUG)


def extract_key_info(processed_results):
    sorted_results = sorted(processed_results, key=lambda x: (x['box'][1], x['box'][0]))

    key_info = {}
    current_text = ""
    matched_keys = set()

    for res in sorted_results:
        text = res['text'].strip()
        if not text: continue

        if '授权方：' in text and '授权方' not in matched_keys:
            start_idx = text.index('授权方：') + 4
            key_info['授权方'] = text[start_idx:].strip()
            matched_keys.add('授权方')
            continue

        if '被授权方：' in text and '被授权方' not in matched_keys:
            start_idx = text.index('被授权方：') + 5
            key_info['被授权方'] = text[start_idx:].strip()
            matched_keys.add('被授权方')
            continue

        period_match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)至(\d{4}年\d{1,2}月\d{1,2}日)', text)
        if period_match and '授权期限' not in matched_keys:
            key_info['授权期限'] = f"{period_match.group(1)}至{period_match.group(2)}"
            matched_keys.add('授权期限')

        if '查询授权方' in text or '用电信息' in text:
            current_text += text
            if '月的用电信息' in current_text and '查询时间段' not in matched_keys:
                month_match = re.search(r'(\d{4}年\d{1,2}月至\d{4}年\d{1,2}月)', current_text)
                if month_match:
                    key_info['查询时间段'] = month_match.group(1)
                    matched_keys.add('查询时间段')
                current_text = ""

        if '至' in text and '授权期限' not in key_info:
            prev_res = sorted_results[sorted_results.index(res)-1] if sorted_results.index(res)>0 else None
            if prev_res and abs(res['box'][1] - prev_res['box'][3]) < 10:
                combined_text = prev_res['text'] + text
                period_match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)至(\d{4}年\d{1,2}月\d{1,2}日)', combined_text)
                if period_match:
                    key_info['授权期限'] = f"{period_match.group(1)}至{period_match.group(2)}"
                    matched_keys.add('授权期限')

    return key_info


def extract_text_from_pdf(pdf_path, ocr, pages_to_process, threshold=0.8):
    result = ocr.ocr(pdf_path, cls=True)
    processed_results = []
    for page_idx, page_result in enumerate(result):
        if page_idx >= pages_to_process:
            break
        if page_result is None:
            logging.debug(f"Empty page {page_idx + 1} detected, skip it.")
            continue
        for detection in page_result:
            box, (text, score) = detection
            if score < threshold:
                continue
            flat_box = [coordinate for point in box for coordinate in point]
            result_dict = {'box': flat_box, 'text': text}
            # result_dict = {
            #     'box': box,
            #     'text': text
            # }
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


def process_pdf_text(pdf_path):
    with fitz.open(pdf_path) as pdf:
        total_pages = pdf.page_count
    # pages_to_process = total_pages
    pages_to_process = total_pages - 1    # 不处理最后一页

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=pages_to_process, use_gpu=0)
    processed_results = extract_text_from_pdf(pdf_path, ocr, pages_to_process, threshold=0.8)

    imgs = []
    with fitz.open(pdf_path) as pdf:
        for pg in range(pages_to_process):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)

    # save_ocr_results(result=ocr.ocr(pdf_path, cls=True), imgs=imgs, output_dir=output_dir)
    key_info = extract_key_info(processed_results)
    print(key_info)
    return processed_results


if __name__ == '__main__':
    input_file = "../input/pdf/xxx.pdf"
    output_base_dir = '../output/contract/'

    pdf_filename = os.path.basename(input_file)
    pdf_output_dir = os.path.join(output_base_dir, os.path.splitext(pdf_filename)[0])
    os.makedirs(pdf_output_dir, exist_ok=True)

    print(f"Processing: {input_file}")
    process_pdf_text(input_file)
