import os
import json
from difflib import SequenceMatcher

import cv2
import numpy as np
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


def full_image_enhancement(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.2
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.11, 0, 255)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return enhanced


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


def process_all_images(input_demo_folder, output_json_folder, pipelines):
    image_files = [f for f in os.listdir(input_demo_folder)
                   if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print(f"Can not found images in {input_demo_folder}")
        return

    for image_file in image_files:
        try:
            img_path = os.path.join(input_demo_folder, image_file)
            enhanced_img = full_image_enhancement(img_path)

            enhanced_file_name = f"enhanced_{image_file}"
            enhanced_path = os.path.join(output_json_folder, 'processed_images', enhanced_file_name)
            cv2.imwrite(enhanced_path, cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))

            output = pipelines.predict(enhanced_path)
            json_file_path = os.path.join(output_json_folder,
                                          f"{os.path.splitext(image_file)[0]}_res.json")

            for res in output:
                res.print()
                res.save_to_img(save_path="../output/seal")
                res.save_to_json(save_path=json_file_path)

            process_json(json_file_path)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue


if __name__ == '__main__':
    input_folder = "../input/demo"
    json_folder_path = "../output/seal/json"
    create_output_folder()

    # pipeline = create_pipeline(pipeline="../config/layout_parsing_v2.yaml")
    pipeline = create_pipeline(pipeline="layout_parsing_v2")
    process_all_images(input_folder, json_folder_path, pipeline)
