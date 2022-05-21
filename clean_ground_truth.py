import scipy.io
import numpy as np
import os 
import json

validation_mat = scipy.io.loadmat("imagenet_sources/meta.mat")["synsets"]
validation_label = list(map(lambda x: x[0][0][0][0], validation_mat))
validation_names = list(map(lambda x: x[0][1][0], validation_mat))
label2name = dict(zip(validation_label, validation_names))
name2label = dict(zip(validation_names, validation_label))
           
with open("imagenet_sources/ILSVRC2012_validation_ground_truth.txt") as fp:
        line = fp.read().splitlines()
        val_gt = np.array(list(map(lambda x: label2name[int(x)], line)))

validation_folder = "/Users/shawin/Downloads/ILSVRC2012_img_val"
validation_image_names = np.array(os.listdir(validation_folder))
validation_image_names.sort()
validation_df = np.stack((validation_image_names, val_gt), axis=1)

with open('imagenet_sources/imagenet_class_index.json') as f:
  data = json.load(f)

data_dict={}
for i in data:
    data_dict[data[i][0]]=int(i)


validation_dict ={}

for i in validation_df:
    validation_dict[i[0]]=data_dict[i[1]]


print(sorted(np.unique(list(validation_dict.values()))))
# with open('real_ground_truth.txt','w') as f:
#     for i in validation_dict.values():
#         f.write(str(i))
#         f.write('\n')
#     f.close()