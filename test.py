import pandas as pd

data = pd.read_csv("data/idimage_fixed.csv")
# data2 = pd.read_csv("data/idmeta.csv")
# data3 = pd.read_csv("data/idlabel.csv")
data3 = pd.read_csv("data/idimage.csv")

# print("fixed")
# print(data3.columns)
# print(data3.head())

# print("original")

# print(data.columns)
# print(data.head())


diff = data.compare(data3)
print(diff)