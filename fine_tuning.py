import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0, 1, 2"
import sys
sys.path = ['', '/usr/local/packages/python/modules/matplotlib-3.1.1/lib/python3.6/site-packages', '/usr/local/packages/python/modules/opencv-4.0.0/lib/python3.6/site-packages', '/usr/local/packages/python/modules/mxnet-1.2.0/lib/python3.6/site-packages', '/usr/local/packages/python/modules/pytorch-0.4.1/lib/python3.6/site-packages', '/usr/local/packages/python/modules/tensorflow-2.2/lib/python3.6/site-packages', '/usr/local/software/python3/lib/python36.zip', '/usr/local/software/python3/lib/python3.6', '/usr/local/software/python3/lib/python3.6/lib-dynload', '/data/yqiuau/qiuyaowen/lib/python3.6/site-packages']
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import load_model

if __name__ == '__main__':

    ########Define Fixed Parameters###############
    epochs = 50
    train_batch_size = 128
    val_batch_size = 128
    input_shape = (224,224)
    nums_of_classes = 1000
    optimizer = SGD(lr=0.0001, momentum=0.8)

    path = 'Model/Baseline/Best_model_38_0.6062.hdf5'
    
    ########Define Model#################
    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1","GPU:2"])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = load_model(path)
        model.trainable = False
        model.layers[-1].trainable = True
        model.layers[-2].trainable = True
        model.layers[-3].trainable = True
        model.layers[-4].trainable = True
    

    #######Training############
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
                                    '/4tssd/imagenet/85_train/',
                                    target_size=input_shape,
                                    batch_size=train_batch_size,)
    val_generator = val_datagen.flow_from_directory(
                                    '/4tssd/imagenet/val/',
                                    target_size=input_shape,
                                    batch_size=val_batch_size,)

    history = model.fit(x = train_generator, 
            epochs=epochs, verbose=1,max_queue_size=40,
            workers=20)



    ########Save the final Model#############
    model.save('Model/FineTune/Final_Model.h5')

    loss, acc1,acc5 = model.evaluate(x = val_generator,max_queue_size=40,workers=20)
    print("loss :{}, acc:{}, acc:{}".format(loss,acc1,acc5))