"""
This script evaluate the performance of model on:
1. Top-5 Accuracy on validation set 1 each epoch
2. Top-1 / Top-5 Accuracy on validation set 2 each epoch
And output the prediction of validation set 1/set 2
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0, 1, 2"
import sys
sys.path = ['', '/usr/local/packages/python/modules/matplotlib-3.1.1/lib/python3.6/site-packages', '/usr/local/packages/python/modules/opencv-4.0.0/lib/python3.6/site-packages', '/usr/local/packages/python/modules/mxnet-1.2.0/lib/python3.6/site-packages', '/usr/local/packages/python/modules/pytorch-0.4.1/lib/python3.6/site-packages', '/usr/local/packages/python/modules/tensorflow-2.2/lib/python3.6/site-packages', '/usr/local/software/python3/lib/python36.zip', '/usr/local/software/python3/lib/python3.6', '/usr/local/software/python3/lib/python3.6/lib-dynload', '/data/yqiuau/qiuyaowen/lib/python3.6/site-packages']
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

path_val2 = "/4tssd/imagenet/val2/"
epochs = 50
batch_size = 128
input_shape = (224,224)


val2_datagen = ImageDataGenerator(rescale=1./255)

val2_generator = val2_datagen.flow_from_directory(
                                path_val2,
                                target_size=input_shape,
                                batch_size=batch_size)

#########Evaluate the Base line Model###############
path = "Model/FineTune/"
eps = range(epochs)
loss = [0]*epochs
top_1_acc_val2 = [0]*epochs
top_5_acc_val2 = [0]*epochs

for file in os.listdir(path):
# if 'Best' in file:
    # epoch = int(file.split('_')[2])
    # print("**********Evaluating Epoch {}*************".format(epoch))
    model = load_model(path+'Final_Model.h5')

    ####evaluate on validation set 2#####
    loss2, acc1_2,acc5_2 = model.evaluate(x = val2_generator,max_queue_size=40,workers=20)
    print("loss 2:{}, acc1:{}, acc2:{}".format(loss2,acc1_2,acc5_2))


    # loss[epoch-1] = loss2
    # top_1_acc_val2[epoch-1] = acc1_2
    # top_5_acc_val2[epoch-1] = acc5_2

# pd.DataFrame({'Epochs':eps,'Loss':loss,
            # 'Top_1_val_2':top_1_acc_val2,'Top_5_val_2':top_5_acc_val2}).to_csv('logs/ft_val2_performance.csv',index=None)