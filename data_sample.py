"""
This script inserts N randomly chosen img in each label (where N = 
the number of pruned example),to maintain the balance between labels.
"""
import os
import pandas as pd
import shutil
import numpy as np

path = '/4tssd/imagenet/TVL/'

pruned_label_count = pd.read_csv('data/_PRUNED_COUNT_TVL.csv')

for i in range(pruned_label_count.shape[0]):
    label = pruned_label_count.iloc[i]['LABEL']
    n = pruned_label_count.iloc[i]['PRUNED_COUNT']

    imgs = os.listdir(path+label+'/')
    random_chosen = np.random.choice(imgs,size=n,replace=True)
    for each in random_chosen:
        # print(path+label+'/'+each,path+label+'/copy_'+each)
        shutil.copyfile(path+label+'/'+each,path+label+'/copy_'+str(np.random.randint(0,10000))+'_'+each)

