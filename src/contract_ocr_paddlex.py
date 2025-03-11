import json
import os
from difflib import SequenceMatcher

import fitz
from paddlex import create_pipeline


def create_output_folder():
    folders = [
        '../output',
        '../output/seal',
        '../output/seal/json',
        '../output/seal/json/processed_images'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    return folders[0]

def process_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    parsing_res_list = data.get('parsing_res_list', [])

    texts = [item['block_content'] for item in parsing_res_list if item['block_label'] == 'text']
    seals = [item['block_content'] for item in parsing_res_list if item['block_label'] == 'seal']

    if not texts and not seals:
        print(f"No valid text or seal content found in {json_file_path}. Skipping processing.")
        return

    best_matches = []
    for seal in seals:
        best_match_text = None
        highest_ratio = 0
        for text in texts:
            ratio = SequenceMatcher(None, seal, text).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match_text = text
        best_matches.append(
            {'seal': seal, 'best_match_text': best_match_text, 'match_ratio': highest_ratio})

    output_file_path = json_file_path.replace('_res.json', '_processed_res.json')
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(best_matches, file, ensure_ascii=False, indent=4)


def process_pdf(pdf_path, output_json_folder, pipelines):
    doc = fitz.open(pdf_path)
    last_page = doc.load_page(doc.page_count - 1)
    pix = last_page.get_pixmap()
    output_image_path = os.path.join(output_json_folder, 'processed_images', 'last_page.png')
    pix.save(output_image_path)

    output = pipelines.predict(output_image_path)
    json_file_path = os.path.join(output_json_folder, f"last_page_res.json")

    for res in output:
        res.print()
        res.save_to_img(save_path="../output/seal")
        res.save_to_json(save_path=json_file_path)

    process_json(json_file_path)


def process_all_images(input_demo_folder, output_json_folder, pipelines):
    files = [f for f in os.listdir(input_demo_folder)
             if f.endswith('.pdf')]

    if not files:
        print(f"Can not found pdfs in {input_demo_folder}")
        return

    for file in files:
        try:
            file_path = os.path.join(input_demo_folder, file)
            if file.endswith('.pdf'):
                process_pdf(file_path, output_json_folder, pipelines)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue


if __name__ == '__main__':
    input_folder = "../input/pdf"
    json_folder_path = "../output/seal/json"
    create_output_folder()

    pipeline = create_pipeline(pipeline="../config/layout_parsing_v2.yaml")
    # pipeline = create_pipeline(pipeline="layout_parsing_v2")
    process_all_images(input_folder, json_folder_path, pipeline)
