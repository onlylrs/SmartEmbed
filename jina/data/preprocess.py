import json
import os
from tqdm import tqdm
from jina.utils.qwen_utils.data_utils import load_image

def convert_to_contrastive_format(
    input_path: str,
    output_path: str,
    image_dir: str = None,
    text_field: str = "text",
    image_field: str = "image"
):
    """
    将标准数据集转换为对比学习格式
    """
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in tqdm(f_in):
            item = json.loads(line)
            
            # 处理文本
            if text_field in item:
                text = item[text_field]
                f_out.write(json.dumps({"text": text}) + '\\n')
            
            # 处理图像（如果有）
            if image_field in item and image_dir:
                image_path = os.path.join(image_dir, item[image_field])
                try:
                    # 验证图像是否存在
                    load_image(image_path)
                    f_out.write(json.dumps({"image": image_path}) + '\\n')
                except:
                    print(f"Skipping invalid image: {image_path}")

if __name__ == "__main__":
    # 示例用法
    convert_to_contrastive_format(
        input_path="data/raw/multilingual_corpus.jsonl",
        output_path="data/processed/contrastive_data.jsonl",
        image_dir="data/images"
    )
