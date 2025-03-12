"""
使用PaddleX识别pdf合同文件最后一页的印章部分
"""
import difflib
import os

import fitz
from paddlex import create_pipeline

def process_pdf(pdf_path, output_json_folder, pipelines):
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
    for seal_text in seal_texts:
        best_match = None
        highest_similarity = 0
        for text_text in text_texts:
            similarity = difflib.SequenceMatcher(None, seal_text, text_text).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = text_text
        print(f"{{{best_match}}}")

    os.remove(output_image_path)


if __name__ == '__main__':
    input_file = "../input/pdf/xxx.pdf"
    # input_url = "https://xxx.pdf"
    json_folder_path = "../output"

    pipeline = create_pipeline(pipeline="../config/layout_parsing_v2.yaml")
    # pipeline = create_pipeline(pipeline="layout_parsing_v2")
    process_pdf(input_file, json_folder_path, pipeline)
