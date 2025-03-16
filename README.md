# VLM-related Issues

## LLaVA Experiment

### Task Target

1. 使用LLaVA v1.5針對各種圖表進行處理及分析，觀察LLaVA v1.5目前所遇到的問題
2. 透過Prompt設計或其他方法引導LLaVA模型辨識出影像中可能存在的命名實體
3. 透過獲取的實體關係結合知識圖譜以加強模型分析及回答問題之能力


## LLaVA Install

If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/waynechang95/VLMEXP.git
cd VLMEXP/LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Dataset

Image segment from real financial statement report including English version and Chinese version.

1. English Version ( Ex: Dataset/English/1.jpg )

    <img src="Dataset/English/1.jpg" width="70%">

2. Chinese Version ( Ex: Dataset/Chinese/1.jpg )

    <img src="Dataset/Chinese/1.jpg" width="70%">


## Run 

Automatic script to collect the output from the model.

1. Setting Path and Prompt ( ex: RUN/test.py )

```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_output import eval_model

import json
import os

model_path = "liuhaotian/llava-v1.5-7b"

# Prompt Design
prompt = "for the following graph, please show what is the graph about?"

# Setting File Path
image_file_path = 'Dataset/English'
image_file_list = os.listdir(image_file_path)

# Setting Output Path
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


```

## Result

1. Output Format

```JSON
{
        "Image": "file path",
        "Answer": "LLaVA output"
    }
```

2. Example (Chinese/1.jpg)

<img src="Dataset/English/1.jpg" width="70%">

```JSON
{
        "Image": "Chinese/Dataset/Chinese/1.jpg",
        "Answer": "The graph is a financial spreadsheet displaying various financial data, including numbers and calculations. It appears to be a table with columns and rows, possibly showing the balance of a company or an individual's financial situation. The numbers are presented in both English and Chinese, indicating that the data might be related to an international or multicultural context. The spreadsheet is filled with numbers and calculations, providing a detailed view of the financial information."
    }

```
