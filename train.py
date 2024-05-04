#####################################################################################
#                                FILE INFORMATION                                   #
#   Filename: train.py                                                              #
#   Main Author:  Ajay T Shaju                                                      #
#   Co-Authors: Justin Thomas Jo, Emil Saj Abraham, Vishnuprasad KG                 #
#   Date(Last Updated): 01/05/2024                                                  #
#####################################################################################

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, TimeDistributed,
                                     LSTM, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Layer, Concatenate, Dropout, Layer, Multiply, Flatten)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler

mixed_precision.set_global_policy('float32')  # for computational speedup by performing operations in half-precision format

res_path = r"C:\Users\ajayt\Project\Results\train" # path to save results
data_dir = r"C:\Users\ajayt\Project\HockeyFight" # path to dataset directory
data_folders = ['train', 'val']
train_dir = os.path.join(data_dir, data_folders[0])  # 'train' folder
val_dir = os.path.join(data_dir, data_folders[1])  # 'val' folder
print(f"Train Path: {train_dir}\nValidation Path: {val_dir}")

# Configure TensorFlow to use GPU memory incrementally
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initialized

# Parameters
batch = 4
num_frames = 20
frame_height = 224
frame_width = 224
num_channels = 3
num_classes = 2
lr_rate = 0.001
loss_fn = 'categorical_crossentropy'

class FramesSequence(Sequence):
    def __init__(self, directory, batch_size=1, shuffle=True, data_augmentation=True):
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path, self.Y_dict = self.search_data()
        # Print basic statistics information
        self.print_stats()
        return None

    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        one_hots = to_categorical(range(len(self.dirs)))
        for i,folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory,folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path,file)
                # append the each file path, and keep its label
                X_path.append(file_path)
                Y_dict[file_path] = one_hots[i]
        return X_path, Y_dict

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files from {} class.".format(self.n_files,self.n_classes))
        for i,label in enumerate(self.dirs):
            print('%10s : '%(label),i)
        return None

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch"""
        # get the indexs of each batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y
    
    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video    
    
    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data-mean) / std
    
    def uniform_sampling(self, video, target_frames=num_frames):
        """Taking a sample of frames from all available frames"""
        # get total frames of input video and calculate sampling interval
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames/target_frames))
        # init empty list for sampled video and
        sampled_video = []
        for i in range(0,len_frames,interval):
            sampled_video.append(video[i])
        # calculate numer of padded frames and fix it
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad>0:
            for i in range(-num_pad,0):
                try:
                    padding.append(video[i])
                except:
                    padding.append(video[0])
            sampled_video += padding
        # get sampled video
        return np.array(sampled_video, dtype=np.float32)
    
    def load_data(self, path):
        data = np.load(path, mmap_mode='r')
        data = np.float32(data)
        if self.data_aug:
            data = self.random_flip(data, prob=0.5)
        data = self.uniform_sampling(video=data, target_frames=num_frames)
        data[...,:3] = self.normalize(data[...,:3])
        return data  
    
class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d = Conv2D(1, (7, 7), activation='sigmoid', padding='same') # used to learn the spatial attention weights
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True) # taking max value of feature map
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True) # taking avg(mean) value of feature map
        concat = Concatenate(axis=-1)([max_pool, avg_pool]) # concating max and avg value along feature map(last channel) axis
        attention = self.conv2d(concat) # initialized before in build method, outputs single act-map having values between 0 and 1
        return Multiply()([inputs, attention]) # multiplying the feature map with the original input tensor element-wise
        # The multiplication is a gating mechanism where values close to 1 in the attention map allow the corresponding features in the input tensor to pass, and values close to 0 will be blocked.

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        return config

def violence_detection_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape[1:])
    
    # Unfreeze the top layers of the model
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Freeze all but the last 20 layers
        layer.trainable = False
    # TimeDistributed layers do the corresponding operations to all the frames passed to the network in a batch
    x = TimeDistributed(base_model)(inputs)
    x = TimeDistributed(SpatialAttention())(x) # to put attention on extracted spatial features
    x = TimeDistributed(BatchNormalization())(x) # to stablize the learned features
    # pooling layers to reduce the number of parameters and spatial dimension
    avg_pool = TimeDistributed(GlobalAveragePooling2D())(x) # takes average of a feature map
    max_pool = TimeDistributed(GlobalMaxPooling2D())(x) # take max value of a feature map
    concatenated_pools = Concatenate(axis=-1)([avg_pool, max_pool]) # normal concatenation
    x = LSTM(128, return_sequences=False)(concatenated_pools) # to capture temporal dynamics
    x = Flatten()(x) # from 2D feature map to 1D vector, for making input dim compatible with Dense layers
    x = Dropout(0.5)(x) # drop off less important features 
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # removing inactive neurons which prevents overfitting
    x = Dense(25, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
model = violence_detection_model((num_frames, frame_height, frame_width, num_channels), num_classes)
model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])
model.summary()

# Check if the checkpoint exists
checkpoint_path = os.path.join(res_path, "Best_Model.keras")
if os.path.exists(checkpoint_path):
    # Load the model with custom objects
    model = load_model(checkpoint_path, custom_objects = {'SpatialAttention': SpatialAttention})
    print("Resuming from the last checkpoint.")
else:
    # If checkpoint does not exist, initialize a new model
    input_shape = (num_frames, frame_height, frame_width, num_channels)
    model = violence_detection_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=lr_rate), loss=loss_fn, metrics=['accuracy'])
    print("Starting training from scratch.")

# Callbacks
checkpoint_path = os.path.join(res_path, 'Best_Model.keras')
checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001, verbose=1)
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
csv_logger = CSVLogger(os.path.join(res_path, 'training_log.csv'), append=True)

# Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
lr_scheduler = LearningRateScheduler(scheduler)

# Data Generators
train_gen = FramesSequence(directory=train_dir, batch_size=batch, shuffle=True, data_augmentation=True)
val_gen = FramesSequence(directory=val_dir, batch_size=batch, shuffle=False, data_augmentation=False)

# Fit the model
# with tf.device('CPU'): # if tf didn't work with GPU
history = model.fit(train_gen, validation_data=val_gen, epochs=100,
                    callbacks=[checkpoint_cb, reduce_lr_cb, early_stopping_cb, csv_logger, lr_scheduler])

# Save the model and training history
model.save(os.path.join(res_path, 'Final_Model.keras'))
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(res_path, 'training_history.csv'), index=False)