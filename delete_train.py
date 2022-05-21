"""
This File will delete image with perpelxity > threshold
"""
import os
import pandas as pd

full_train = pd.read_csv('data/FULL_TRAIN.csv')
remain_train = pd.read_csv('data/_PRUNED_TRAINING_TVL.csv')

delete_term = set(full_train['PATH']).difference(set(remain_train['PATH']))

original_path = 'train'
pruned_path = 'TVL'
for term in delete_term:
    true_path = term.replace(original_path,pruned_path)
    os.remove(true_path)
    