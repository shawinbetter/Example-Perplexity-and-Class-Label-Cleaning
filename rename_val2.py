import os
import pandas as pd

mapper = pd.read_csv('data/id_label_map.csv')

id_folder= dict(zip(mapper['ID'],mapper['Nums']))

path = "/4tssd/imagenet/val2/"

for folder in os.listdir(path):
    # print(folder,id_folder[int(folder)])
    os.rename(path+folder,path+id_folder[int(folder)])