import os

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


def process_pdf(pdf_path, output_json_folder, pipelines):
    doc = fitz.open(pdf_path)
    last_page = doc.load_page(doc.page_count - 1)
    pix = last_page.get_pixmap()
    output_image_path = os.path.join(output_json_folder, 'processed_images', 'last_page.png')
    pix.save(output_image_path)

    output = pipelines.predict(output_image_path)

    for res in output:
        prediction_result = res.json
        parsing_res_list = prediction_result.get('res', {}).get('parsing_res_list', [])
        for block in parsing_res_list:
            if block.get('block_label') == 'seal':
                block_content = block.get('block_content')
                # block_bbox = block.get('block_bbox')
                # print(f"{{'box': '{block_bbox}', 'text': '{block_content}'}}")
                print(f"{{'text': '{block_content}'}}")
    os.remove(output_image_path)


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
