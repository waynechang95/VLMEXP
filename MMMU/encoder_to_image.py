import base64
import numpy as np
import pandas as pd

from io import BytesIO
from PIL import Image

import json
import os

input_path = 'MMMU/Accounting/validation-00000-of-00001.parquet'

mmmu = pd.read_parquet(
    input_path, engine='pyarrow')

mmmu_png = mmmu['image_1']

for i in range(len(mmmu_png)):
    image_data = np.array(mmmu['image_1'][i]['bytes'])

    # Encode the image data into Base64
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    decoded_bytes = base64.b64decode(encoded_image)

    image = Image.open(BytesIO(decoded_bytes))

    if i < 10:
        output_path = 'MMMU/Accounting/image00' + str(i) + '.png'
    elif i < 100:
        output_path = 'MMMU/Accounting/image0' + str(i) + '.png'
    else:
        output_path = 'MMMU/Accounting/image' + str(i) + '.png'
    image.save(output_path)
