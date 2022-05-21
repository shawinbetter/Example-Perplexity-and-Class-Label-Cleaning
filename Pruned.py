import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0, 1, 2"
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy


if __name__ == '__main__':

    ########Define Fixed Parameters###############
    epochs = 50
    train_batch_size = 128
    val_batch_size = 128
    input_shape = (224,224)
    nums_of_classes = 1000
    optimizer = SGD(lr=0.1, momentum=0.8)

    ########Define Model#################
    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1","GPU:2"])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = MobileNetV2(weights=None)
        model.compile(loss='categorical_crossentropy', metrics=['acc',TopKCategoricalAccuracy(k=5)],optimizer=optimizer)
    

    ########Define Model Callback#################
    filepath="Best_model_{epoch:02d}_{val_acc:.4f}.hdf5"

    checkpoint = ModelCheckpoint(filepath = 'Model/Pruned85/'+filepath, monitor='val_acc',verbose=1,save_best_only=False)
    
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.1
        epochs_drop = 10
        end_lr = 0.0001
        lrate = initial_lrate * np.power(drop,  
            np.floor((1+epoch)/epochs_drop))
        if lrate > end_lr:
            return lrate
        else:
            return end_lr
    lr_scheduler = LearningRateScheduler(step_decay)

    csv_logger = CSVLogger('logs/85_pruned.log')

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
            validation_data = val_generator, 
            epochs=epochs, verbose=1,max_queue_size=40,
            workers=20,
            callbacks=[checkpoint,lr_scheduler, csv_logger])


    ########Save the final Model#############
    model.save('Model/Pruned85/Final_Model.h5')