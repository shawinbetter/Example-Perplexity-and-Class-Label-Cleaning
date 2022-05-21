import os
import shutil
import pandas as pd

mapper = pd.read_csv('data/id_label_map.csv')

id_folder= dict(zip(mapper['ID'],mapper['Nums']))

path = '/4tssd/imagenet/val/'

with open('imagenet_sources/real_ground_truth.txt') as f:
    lines = f.readlines()
lines = [int(i.replace('\n',"")) for i in lines]
dic = dict(zip(range(1,len(lines)+1),lines)) #{row:label_num}

count = 0
for jpg in os.listdir(path): #for jpg file
    ID = int(jpg.split('_')[2].split('.')[0]) #ID is row number
    label_num = dic[ID]  #get the true label range[0,999]

    folder = id_folder[label_num] #find the real folder the jpg belogns to


    if not os.path.exists(path+folder):
        os.mkdir(path+folder)
    shutil.move(path+jpg,path+folder)


