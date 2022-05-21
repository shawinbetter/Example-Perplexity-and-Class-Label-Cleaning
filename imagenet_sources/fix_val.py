import os
import pandas as pd

path = '/4tssd/imagenet/val/'

with open('/data/yqiuau/Indepedent_Project/imagenet_sources/ILSVRC2012_validation_ground_truth.txt') as f:
    lines = f.readlines()
lines = [int(i.replace('\n',"")) for i in lines]

dic = dict(zip(range(1,len(lines)+1),lines))

mapper = pd.read_csv('data/id_label_map.csv')
id_nums = dict(zip(mapper['ID'],mapper['Nums']))

for each in os.listdir(path):
    first_img_name = os.listdir(path+each+'/')[0]
    ID = int(first_img_name.split('_')[2].split('.')[0])
    label_num = dic[ID]
    new_name = id_nums[label_num]
    os.rename(path+each,path+new_name+'_tmp')

for each in os.listdir(path):
    os.rename(path+each,(path+each).replace('_tmp',''))
