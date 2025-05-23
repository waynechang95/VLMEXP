from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_output import eval_model

import json
import os

model_path = "liuhaotian/llava-v1.5-7b"

prompt = "for the following graph, please show what is the graph about?"

image_file_path = 'Dataset/English'
image_file_list = os.listdir(image_file_path)

output_path = 'Result/output.json'

output = []

for i in image_file_list:

    image_file = image_file_path + i

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 4096
    })()

    answer = {
        "Image": image_file,
        "Answer": eval_model(args)
    }

    output.append(answer)

with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(output, outfile, ensure_ascii=False, indent=4)
