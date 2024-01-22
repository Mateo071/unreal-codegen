import os
import pandas as pd
current_directory = os.getcwd()
print(current_directory)

filename = 'codegen25/train-00000-of-00001-d9b93805488c263e.parquet'
df = pd.read_parquet(filename)
print(df.head)
