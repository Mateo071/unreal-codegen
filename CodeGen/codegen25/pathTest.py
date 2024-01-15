# import os

# # Get the list of all files and directories
# # in the current working directory
# dir_list = os.listdir()

# print("Files and directories in '", os.getcwd(), "' :")

# # print the list
# print(dir_list)



import pandas as pd
filename = 'codegen25/train-00000-of-00001-d9b93805488c263e.parquet'

# load into a data frame
df = pd.read_parquet(filename)
print(df.head)