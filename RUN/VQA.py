from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_output import eval_model

import pandas as pd
import json
import os

input_path = 'MMMU/Accounting/validation-00000-of-00001.parquet'
output_path = 'Result/VQA/output.json'

mmmu = pd.read_parquet(
    input_path, engine='pyarrow')

output = []


for i in range(30):
    if i < 10:
        image_path = 'MMMU/Accounting/image00' + str(i) + '.png'
    elif i < 100:
        image_path = 'MMMU/Accounting/image0' + str(i) + '.png'
    else:
        image_path = 'MMMU/Accounting/image' + str(i) + '.png'

    args = type('Args', (), {
        "model_path": "liuhaotian/llava-v1.6-vicuna-7b",
        "model_base": None,
        "model_name": "llava-v1.6-vicuna-7b",
        "query": mmmu['question'][i]+mmmu['options'][i],
        "conv_mode": None,
        "image_file": image_path,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 4096
    })()

    answer = {
        "Question": mmmu['question'][i],
        "Options": mmmu['options'][i],
        "Image": image_path,
        "Answer": eval_model(args),
        "Explanation": mmmu['explanation'][i],
        "Ref_Answer": mmmu['answer'][i],
        "Subfield": mmmu['subfield'][i],
        "Topic_difficulty": mmmu['topic_difficulty'][i],
        "Question_type": mmmu['question_type'][i],

    }

    output.append(answer)

with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(output, outfile, ensure_ascii=False, indent=4)
