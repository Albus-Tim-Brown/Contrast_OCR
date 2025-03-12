"""
TODO 完全使用PaddleX通用版面解析产线识别整个合同文件
缺陷：带下划线文本识别异常
"""
from paddlex import create_pipeline

if __name__ == '__main__':
    input_folder = "../input/pdf/xxx.pdf"
    # input_url = "https://xxx.pdf"
    output_folder = "../output/contract"
    # create_output_folder()

    pipeline = create_pipeline(pipeline="../config/layout_parsing_v2.yaml")
    output = pipeline.predict(input=input_folder)
    # output = pipeline.predict(input=input_url)

    for res in output:
        prediction_result = res.json
        # TODO 结果处理

        res.save_to_json(output_folder)
        res.save_to_img(output_folder)
        print(prediction_result)
