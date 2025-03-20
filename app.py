"""
使用PaddleX识别pdf合同文件，只识别最后一页
"""
import difflib
import json
import os
import re

import fitz
from paddlex import create_pipeline
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/process_pdf', methods=['POST'])
def process_pdf_api():
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Please upload pdf file!"}), 400

    temp_dir = ".\\temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_pdf_path = os.path.join(temp_dir, file.filename)
    file.save(temp_pdf_path)
    result_json = process_pdf_seal(temp_pdf_path, temp_dir, pipeline)
    os.remove(temp_pdf_path)
    return jsonify(json.loads(result_json)), 200


def process_pdf_seal(pdf_path, output_json_folder, pipelines):
    doc = fitz.open(pdf_path)
    last_page = doc.load_page(doc.page_count - 1)
    pix = last_page.get_pixmap()
    output_image_path = os.path.join(output_json_folder, 'seal', 'last_page.png')
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
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


if __name__ == '__main__':
    pipeline = create_pipeline(pipeline="./config/PP-StructureV3.yaml")
    app.run(host='0.0.0.0', port=5000)