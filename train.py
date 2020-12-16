import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.optimizers import Adam
import model_keras
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
import numpy as np

data_gen_call_count = 0

def get_training_data(path_velodyne_2d, batch_index, batch_size=2500):
    """
    Function generates a batch of data as per the batch index and the batch size
    Here batch size is the total number of frames we want to load into memory, model.fit will have its's own batch size
    """

    global data_gen_call_count

    debug = 0

    frames = os.listdir(path_velodyne_2d)  # get a list of all present frames
    num_frames = len(frames)
    print("number of training files : ", num_frames)

    if data_gen_call_count == 0:
        np.random.shuffle(frames)  # shuffle all frames once
        data_gen_call_count = 1

    batch_frame_indices = list(range(batch_index * batch_size, batch_size +  (batch_size * batch_index)))  # get a list of indices for frames to be returned in this batch, index is not same as frame number

    train_x_normalized = np.zeros((batch_size, 64, 256, 1))  # to store outgoing data
    train_y_kp = np.zeros((batch_size, 64, 256))
    
    for index, frame_index in enumerate(batch_frame_indices):
        f_num = frames[frame_index]  # get the actual frame number from the index
        frame_path = path_velodyne_2d + '/' + str(f_num)  # create full path
        file = np.load(frame_path)  # read the frame
        file = file[0:64, 0:256, :]  # get 64, 256 shape
        
        train_x_normalized[index] = np.expand_dims(file[:, :, 3], -1)  # get the depth channel only
        train_x_normalized[index] /= np.max(train_x_normalized[index])  # normalize by the max value

        if debug:
            plt.imshow(train_x_normalized[index])
            plt.show()

        train_y_kp[index] = file[:, :, 5]  # get the key points

        if debug:
            plt.imshow(train_y_kp[index])
            plt.show()
    
    return train_x_normalized, train_y_kp


def get_train_data_x_y(path_velodyne_train_car):
    """
    Function return data in X, Y tuple
    """

    frames = os.listdir(path_velodyne_train_car)
    np.random.shuffle(frames)
    num_frames = len(os.listdir(path_velodyne_train_car))  # total number of frames

    for i in range(num_frames):
        frames[i] = path_velodyne_train_car + '/' + frames[i]
    
    train_x_normalized = np.zeros((num_frames, 64, 256, 1))
    train_y_kp = np.zeros((num_frames, 64, 256))

    for index, frame in enumerate(frames):
        file = np.load(frame)
        file = file[0:64, 0:256, :]
        train_x_normalized[index] = np.expand_dims(file[:, :, 3], -1)
        train_x_normalized[index] /= np.max(train_x_normalized[index])
        train_y_kp[index] = file[:, :, 5]
    
    return train_x_normalized, train_y_kp


def train_model_epochs(compiled_model, num_epochs=50, batch_size=28):
    """
    Trains the compiled keras model
    """

    path_velodyne_train_car = '/home/data/KITTI/training/velodyne_2d_car'

    train_X, train_Y = get_train_data_x_y(path_velodyne_train_car)  # get the training data

    print(train_X.shape)
    print(train_Y.shape)

    compiled_model.fit(train_X, train_Y, verbose=1, batch_size=batch_size, epochs=num_epochs)
    
    return compiled_model


if __name__ == "__main__":

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    model_unet = model_keras.get_model((64, 256), 2)  # get a model to only detect cars
    model_unet.summary()
    
    optim = tf.keras.optimizers.Adam()
    loss_kp = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # with this labels don't need one-hot encoding
    # loss_kp = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model_unet.compile(optimizer=optim, loss=loss_kp, metrics=['accuracy'])  # compile the model

    trained_model = train_model_epochs(model_unet, 90)

    trained_model.save("trained_u3d_new")
    trained_model.save("trained_u3d_new.h5")

