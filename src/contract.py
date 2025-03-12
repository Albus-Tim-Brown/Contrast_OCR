"""
合同文件识别
"""
import difflib
import json
import os
import re
import logging
import fitz
import fitz
import cv2
import numpy as np

from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
from paddlex import create_pipeline

logging.basicConfig(level=logging.DEBUG)


# ocr seal
def process_pdf_seal(pdf_path, output_json_folder, pipelines):
    doc = fitz.open(pdf_path)
    last_page = doc.load_page(doc.page_count - 1)
    pix = last_page.get_pixmap()
    output_image_path = os.path.join(output_json_folder, 'seal', 'last_page.png')
    pix.save(output_image_path)

    output = pipelines.predict(output_image_path)

    seal_texts = []
    text_texts = []

    for res in output:
        prediction_result = res.json
        parsing_res_list = prediction_result.get('res', {}).get('parsing_res_list', [])
        for block in parsing_res_list:
            if block.get('block_label') == 'seal':
                seal_texts.append(block.get('block_content'))
            if block.get('block_label') == 'text':
                text_texts.append(block.get('block_content'))

    matches = []
    for seal_text in seal_texts:
        best_match = None
        highest_similarity = 0
        for text_text in text_texts:
            similarity = difflib.SequenceMatcher(None, seal_text, text_text).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = text_text
        matches.append(best_match)

    result = {}
    authorizer_pattern = re.compile(r"授权方\s*：\s*(.+)$", re.DOTALL)
    authorized_pattern = re.compile(r"被授权方\s*：\s*(.+)$", re.DOTALL)
    for match in matches:
        authorizer_match = authorizer_pattern.search(match)
        if authorizer_match:
            result["SealAuthorizer"] = authorizer_match.group(1).strip()
        authorized_match = authorized_pattern.search(match)
        if authorized_match:
            result["SealAuthorized"] = authorized_match.group(1).strip()
    json_output = json.dumps(result, ensure_ascii=False, indent=2)

    print(json_output)
    os.remove(output_image_path)
    return json_output


# ocr text
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



# main
if __name__ == '__main__':
    input_file = "../input/pdf/xxx.pdf"
    # input_url = "https://xxx.pdf"
    json_folder_path = "../output"
    output_base_dir = '../output/contract/'

    pdf_filename = os.path.basename(input_file)
    pdf_output_dir = os.path.join(output_base_dir, os.path.splitext(pdf_filename)[0])
    os.makedirs(pdf_output_dir, exist_ok=True)

    print(f"Processing: {input_file}")
    process_pdf_text(input_file)
    pipeline = create_pipeline(pipeline="../config/layout_parsing_v2.yaml")
    # pipeline = create_pipeline(pipeline="layout_parsing_v2")
    json_result = process_pdf_seal(input_file, json_folder_path, pipeline)