# Importing libraries, APIS & Tensorflow

# import future

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow & keras

import tensorflow as tf
import keras

# import modules, libraries, APIs, optimizers, layers, etc...

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    MaxPool1D,
    GlobalAveragePooling2D
)
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras import models
from keras import layers
from keras import optimizers
import scipy
from scipy.interpolate import make_interp_spline
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import PIL
from PIL import Image
from contextlib import redirect_stdout
import sys

# print file name

file_name =  os.path.basename(sys.argv[0])
file_name = str(file_name)

# Trim the ".py" extensiojn from the file 

file_name = file_name[:-3]

print("Name of the File:", file_name)

# set another (very high) limit for image processing

PIL.Image.MAX_IMAGE_PIXELS = 9933120000

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# setting the paths, pathfiles

parent_dir = "/mnt/scratch_b/users/a/agavros/Database/"

train_labels_path = parent_dir + "Labels/" + "new_class_labels.xlsx"
path_train_images = parent_dir
saved_model_path = "/mnt/scratch_b/users/a/agavros/Saved_Models/"
path_scripts = "/mnt/scratch_b/users/a/agavros/Scripts/"

### Setting up all the parameters for the training, processing of data and produced files management ###

# import the labels of the dataset

data = pd.read_excel(train_labels_path, engine="openpyxl")

# check the dataframe to inspect any NaNs or missing values

print(data.head(3))

# check the number of rows and columns of the dataframe

print(data.shape)

### setting parameters for the saving and loading of produced files ###

# set batch size

# num_batch = 10

# adjusting the size of images into the same dimensions

img_width = 250
img_height = 250

### adjsusting training parameters in hidden layers ###

# Batch Normalization application or not

batch_norm = "Yes"

# 1st hidden layer

num_neurons_1st_hidden = 16
drop_perc_1st_hidden = 0.2

# 2nd hidden layer

num_neurons_2nd_hidden = 32
drop_perc_2nd_hidden = 0.3

# 3d hidden layer

num_neurons_3d_hidden = 64
drop_perc_3d_hidden = 0.4

# 4th hidden layer

num_neurons_4th_hidden = 128
drop_perc_4th_hidden = 0.5

# adjsusting training parameters in dense layers

lrn_rate = 0.0003  # learning rate
num_epochs = 30  # number of training epochs
act_func = "sigmoid"  # activation fuction

num_neurons_1st_dense = 500 # number of neurons in first dense layer
drop_perc_1st_dense = 0 # percentage of dropout rate in first dense layer

num_neurons_2nd_dense = 250 # number of neurons in second dense layer
drop_perc_2nd_dense = 0 # percentage of dropout rate in second dense layer

num_neurons_3d_dense = 0 # number of neurons in third dense layer
drop_perc_3d_dense = 0 # percentage of dropout rate in third dense layer

# create an empty list to take in the images

X = []

# attach the images with the labels, in order to have supervised learning and convert them to have values between 0 - 1 by multiplying with 255 (RGB values: 0 - 255)

for i in tqdm(range(data.shape[0])):
    path = (
        path_train_images
        + data["Art_Genre"][i]
        + "/"
        + data["Artist_Name"][i]
        + "/"
        + data["Painting_Name"][i]
        + ".jpg"
    )
    img = image.load_img(path, target_size=(img_width, img_height, 3))
    img = image.img_to_array(img)
    img = img / 255.0
    X.append(img)

# convert images into np array

X = np.array(X)

# get the size of dataframe X

print("The shape of X list is: ", X.shape)

# removing "unnecessary columns, in order to have only the one-hot encoded columns for the training

y = data.drop(["Artist_Name", "Painting_Name", "Art_Genre"], axis=1)
y = np.array(y)

# inspect the size of new labels dataframe and make sure that columns match the number of classes

print("The shape of y list is: ", y.shape)

# inspect the size of new labels dataframe and make sure that columns match the number of classes

print("The shape of y list is: ", y.shape)

# set that the evaluation set is 10% of the total dataset
# the testing dataset will be 20% of the total dataset
# the final percentage of training set will be 70% of the total dataset


# create the train/test sets, set the size of test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create the test/validation sets, set the size of validation set

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)

# get the shapes of train, test and validation splits

print("The shape of X_train is: ", X_train.shape)
print("The shape of y_train is: ", y_train.shape)

print("The shape of X_test is: ", X_test.shape)
print("The shape of y_test is: ", y_test.shape)

print("The shape of X_val is: ", X_val.shape)
print("The shape of y_val is: ", y_val.shape)

# create the convolutional network

# set the model as sequential
# imply batchnormalization in every layer to make training faster
# imply dropout in each hidden layer to avoid overfitting

model = Sequential()

# 1st hidden layer

model.add(Conv2D(num_neurons_1st_hidden, (3, 3), activation="relu", input_shape=X_train[0].shape, padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(drop_perc_1st_hidden))

# 2nd hidden layer

model.add(Conv2D(num_neurons_2nd_hidden, (3, 3), activation="relu", padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(drop_perc_2nd_hidden))

# 3d hidden layer

model.add(Conv2D(num_neurons_3d_hidden, (3, 3), activation="relu", padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(drop_perc_3d_hidden))

# 4th hidden layer

model.add(Conv2D(num_neurons_4th_hidden, (3, 3), activation="relu", padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(drop_perc_4th_hidden))

# flatten all the layers

model.add(Flatten())

# add the dense layer to obtain all the information from the hidden layers

model.add(Dense(num_neurons_1st_dense, activation="relu"))
model.add(Dropout(drop_perc_1st_dense))

# add a 2nd dense layer to get better results

model.add(Dense(num_neurons_2nd_dense, activation="relu"))
model.add(Dropout(drop_perc_2nd_dense))


# add the final, output layer to get the information regarding multi-class classification

model.add(Dense(10, activation=act_func))

# set the optimizer and loss functions to measure and adjust the weighs of parameters in layer for every iteration

Adam = tf.keras.optimizers.Adam(
    learning_rate=lrn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False
)
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

# train the model

history = model.fit(
    X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test)
)

# summarize history for accuracy

scores = model.evaluate(
    X_test,
    y_test,
    verbose=1,
    sample_weight=None,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)

print(scores)

print(history.history.keys())


# plot the accuracy for train/test sets to check for overfitting or underfitting (less possible)
# plot the loss for train/test sets for same reasons

# plot the accuracy for train/test sets to check for overfitting or underfitting (less possible)
# plot the loss for train/test sets for same reasons

# Create smoother lines in the graphs

epoch_list = []

acc_list = history.history["accuracy"]
print('accuracy list:', acc_list)

val_acc_list = history.history["val_accuracy"]
print('validation accuracy list:', val_acc_list)

loss_list = history.history["loss"]
print('loss list:', loss_list)

val_loss_list = history.history["val_loss"]
print('validation loss list:', val_loss_list)

epoch_lim = num_epochs + 1

for i in range(1, epoch_lim):
    epoch_list.append(i)
    
x_plot = np.array(epoch_list)

y_plot_acc = np.array(history.history["accuracy"])
y_plot_val_acc = np.array(history.history["val_accuracy"])

X_Y_Spline_acc = make_interp_spline(x_plot, y_plot_acc)
X_Y_Spline_val_acc = make_interp_spline(x_plot, y_plot_val_acc)


# Returns evenly spaced numbers
# over a specified interval.

X_plot = np.linspace(x_plot.min(), x_plot.max())
y_plot_acc = X_Y_Spline_acc(X_plot)
y_plot_val_acc = X_Y_Spline_val_acc(X_plot)

# Plotting the Graph

plt.plot(X_plot, y_plot_acc)
plt.plot(X_plot, y_plot_val_acc)
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="lower right")
plt.savefig(
    saved_model_path
    + file_name + "_model_accuracy.png"
)

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

y_plot_loss = np.array(history.history["loss"])
y_plot_val_loss = np.array(history.history["val_loss"])

X_Y_Spline_loss = make_interp_spline(x_plot, y_plot_loss)
X_Y_Spline_val_loss = make_interp_spline(x_plot, y_plot_val_loss)

# Returns evenly spaced numbers
# over a specified interval.

y_plot_loss = X_Y_Spline_loss(X_plot)
y_plot_val_loss = X_Y_Spline_val_loss(X_plot)

# Plotting the Graph

plt.plot(X_plot, y_plot_loss)
plt.plot(X_plot, y_plot_val_loss)
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.savefig(
    saved_model_path
    + file_name + "_model_loss.png"
)

print("Train procedure was completed successfully")

# testing of model's predicting ability in unknown sample

print("Testing procedure started")


# make predictions

predictions = model.predict(X_val)

# get the max values of the predictions and true values to get the metric values

predictions = np.argmax(predictions, axis=1)
y_val = np.argmax(y_val, axis=1)

pososto_epityxias = accuracy_score(y_val, predictions)

pososto_epityxias = pososto_epityxias * 100
pososto_epityxias = round(pososto_epityxias, 2)

print("Success rate: {} %".format(pososto_epityxias))


# create a txt file to save the training parameters of the created model

with open(
    (
        saved_model_path
        + file_name + "_configuration.txt"
    ),
    "w",
) as f:
    f.write(
        "Successful identification rate: {} % \nNumber of epochs: {} \nLearning rate: {} \nImage height and width: {} \nActivation function: {} \naccuracy list: {} \nvalidation accuracy list: {} \nloss list: {} \nvalidation loss list: {} \n\n\n\n\n\n\n\n{}".format(
            pososto_epityxias,
            num_epochs,
            lrn_rate,
            img_width,
            act_func,
            acc_list,
            val_acc_list,
            loss_list,
            val_loss_list,
            model.summary(),
        )
    )

# add the model summary to the created txt file

with open(
    (
        saved_model_path
        + file_name + "_configuration.txt"
    ),
    "a",
) as f:
    with redirect_stdout(f):
        model.summary()
        
# construct a confussion matrix to examine in which style the code performed better

conf_mat = confusion_matrix(y_val, predictions)

# get the names of the classes that correspond to the binary data-predictions/true values

cols_list = data['Art_Genre'].unique()
class_names = cols_list

# make a dataframe of the confusion matrix

conf_mat_dataframe = pd.DataFrame(
    conf_mat, index=[i for i in class_names], columns=[i for i in class_names]
)

# plot and save the confusion matrix

plt.figure(figsize=(20, 16))
plt.rcParams.update({"font.size": 18})
plt.rcParams.update({"font.family": "georgia"})
plt.locator_params(axis="both", integer=True, tight=True)
plt.title("Confusion Matrix")
plt.ylabel("Actual Art Classes")
plt.xlabel("Predicted Art Classes")
sn.heatmap(
    conf_mat_dataframe,
    annot=True,
    linewidths=0.1,
    linecolor="g",
    cbar=True,
    cmap="cividis",
)

plt.savefig(
    saved_model_path
    + file_name + "_confusion_matrix.png"
)


# open and set the config file to enter all the training parameters
# and performance results

df_config = pd.read_excel("/mnt/scratch_b/users/a/agavros/Saved_Models/Config_customs.xlsx", engine = "openpyxl")

# set the number of row to enter the inputs from this script

input_placeholder = len(df_config)
input_placeholder = input_placeholder + 1

# enter the name of file in config dataset

df_config.at[input_placeholder, 'File'] = file_name

# enter the accuracy percentage in config dataset

df_config.at[input_placeholder, 'Rate (%)'] = pososto_epityxias

# enter the dimensions of the input images in config dataset

df_config.at[input_placeholder, 'Hight/Width'] = img_width

# enter the epochs number in config dataset

df_config.at[input_placeholder, 'Epochs'] = num_epochs

# enter the learning rate value in config dataset

df_config.at[input_placeholder, 'Learning_Rate'] = lrn_rate

# enter the activation function in config dataset

df_config.at[input_placeholder, 'Act_Func.'] = act_func

# enter if batch normalization is applied 

df_config.at[input_placeholder, 'Batch_Norm.'] = batch_norm

# enter the number of neurons of first hidden layer in config dataset

df_config.at[input_placeholder, 'Neuron_1st_Hidden'] = num_neurons_1st_hidden

# enter the dropout rate of first hidden layer in config dataset

df_config.at[input_placeholder, 'Dropout_1st_Hidden'] = drop_perc_1st_hidden

# enter the number of neurons of second hidden layer in config dataset

df_config.at[input_placeholder, 'Neuron_2nd_Hidden'] = num_neurons_2nd_hidden

# enter the dropout rate of second hidden layer in config dataset

df_config.at[input_placeholder, 'Dropout_2nd_Hidden'] = drop_perc_2nd_hidden

# enter the number of neurons of third hidden layer in config dataset

df_config.at[input_placeholder, 'Neuron_3d_Hidden'] = num_neurons_3d_hidden

# enter the dropout rate of third hidden layer in config dataset

df_config.at[input_placeholder, 'Dropout_3d_Hidden'] = drop_perc_3d_hidden

# enter the number of neurons of fourth hidden layer in config dataset

df_config.at[input_placeholder, 'Neuron_4th_Hidden'] = num_neurons_4th_hidden

# enter the dropout rate of third hidden layer in config dataset

df_config.at[input_placeholder, 'Dropout_4th_Hidden'] = drop_perc_4th_hidden

# enter the number of neurons of first dense layer in config dataset

df_config.at[input_placeholder, 'Neuron_1st_Dense'] = num_neurons_1st_dense

# enter the dropout rate of first dense layer in config dataset

df_config.at[input_placeholder, 'Dropout_1st_Dense'] = drop_perc_1st_dense

# enter the number of neurons of second dense layer in config dataset

df_config.at[input_placeholder, 'Neuron_2nd_Dense'] = num_neurons_2nd_dense

# enter the dropout rate of second dense layer in config dataset

df_config.at[input_placeholder, 'Dropout_2nd_Dense'] = drop_perc_2nd_dense

# enter the number of neurons of third dense layer in config dataset

df_config.at[input_placeholder, 'Neuron_3d_Dense'] = num_neurons_3d_dense

# enter the dropout rate of third dense layer in config dataset

df_config.at[input_placeholder, 'Dropout_3d_Dense'] = drop_perc_3d_dense

# save the config dataset

df_config.to_excel(saved_model_path + "Config.xlsx",
                   engine='openpyxl',
                   index = False)
                   
### Rename the script file to add the accuracy percentage in the file name

file_before = path_scripts + file_name + ".py" 
file_after = path_scripts + file_name + "_(" + str(pososto_epityxias) + ")"  + ".py"

os.rename(file_before, file_after)
                   
print("The script was executed successfully")
