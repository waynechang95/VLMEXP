import numpy as np
import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq

# pq.read_table('dataset_mmmu.parquet').to_pandas()


finance = pd.read_parquet(
    '/mnt/d/NCCUCS/Lab/VLMEXP/MMMU/Accounting/validation-00000-of-00001.parquet', engine='pyarrow')

print(finance.columns)
print(finance['answer'][10])
print(finance['explanation'][10])
