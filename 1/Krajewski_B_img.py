# Solution for task 2 (Image Classifier) of lab assignment for FDA SS23 by [Bartosz Krajewski]
# imports here
import numpy as np   # essential for everything
import pandas as pd   # data structure
import matplotlib.pyplot as plt   # plots
#import seaborn as sns   # fanc plots
import sklearn   # standard machine learning
import keras   # neural networks (frontend for tensorflow)
from keras.models import Sequential  # simplest way to set up neural network model 
from keras.layers import Dense  # fully connected layer
from tensorflow.keras.optimizers import SGD  # stochastic gradient descent
import tensorflow as tf  # backend, we only use it to set random seed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Reshape, Conv2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# define additional functions here


def train_predict(X, Y, X_test):

    # check that the input has the correct shape
    assert X.shape == (len(X), 6336)
    assert Y.shape == (len(Y), 1)
    assert X_test.shape == (len(X_test), 6336)
  
    # add your data preprocessing, model definition, training and prediction between these lines
    X = X / 255.0 ## its better for MLP to have normalized values
    Y = to_categorical(Y)   ## categorizing
    X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size = 0.1) ## i splitted it for the validation set! 
    
    ## X TEST  has to be normalized too ##
    X_test = X_test / 255


    ## MODEL CREATION ##
    drop_rate = 0.5
    input = Input(shape=(6336))
    x = Reshape(target_shape=(44, 48, 3))(input) ## reshape so we get 1 image !
    x = Conv2D(32, kernel_size=3, activation="relu", kernel_initializer="he_normal")(x) ## conv layers 
    x = BatchNormalization()(x) ## Batchnormalization after every layer we normalize input values
    ## I havent used maxpooling because it gives better scores without that!
    x = Conv2D(32, kernel_size=3, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, kernel_size=5, activation="relu", kernel_initializer="he_normal", strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Dropout(drop_rate)(x) ## https://keras.io/api/layers/regularization_layers/dropout/
    ##The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, 
    ##which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.

    x = Conv2D(64, kernel_size=3, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=3, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=5, activation="relu", kernel_initializer="he_normal", strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Dropout(drop_rate)(x)

    x = Conv2D(128, kernel_size=4, activation="relu", kernel_initializer="he_normal")(x) ## in all of conv layers kernel_size is chcanged, so the size of conv window changes!
    x = BatchNormalization()(x)

    x = Dropout(drop_rate)(x)
    ## Flatten receives input after conv layers ##
    x = Flatten()(x)
    x = Dense(1000, activation="relu")(x) ## dense layer of 1000 neurons 
    x = Dropout(drop_rate)(x)
    y = Dense(Y_train.shape[1], activation="softmax")(x) ## softmax with 40 labels - in labels there are probabilities which tells which image is most probable

    model = Model(input, y)
    ## Stop if val_acc is getting lower !! ##
    stop = EarlyStopping(monitor="val_acc", min_delta=0, patience=50, restore_best_weights=True)
    ## TRAIN ## 
    for b, p in [(32, 0.001), (16, 0.0001)]: ## fitting with 2 parameters to adam and 2 batch sizes ! 
        model.compile(Adam(p), "categorical_crossentropy", metrics=["acc"])
        model.fit(X_train, Y_train, batch_size=b, epochs=50, validation_data=(X_test_val, Y_test_val), callbacks=[stop])
        print(f"Val acc: {model.evaluate(X_test_val, Y_test_val)[1]:.04f}") ## checking the model 

    ## predict ##
    y_pred = model.predict(X_test) ## its softmax in the output consisting of 40 layers with probabilieites of labels -- > we have to take maximum value from that
    
    y_pred= np.argmax(y_pred, axis = 1) ## max value which tells us which sign is that 
    # test that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),) or y_pred.shape == (len(X_test), 1) ## its softmax layer with 40 

    return y_pred ## the resulting accuracy value for model.evaluate was ~0.97


if __name__ == "__main__":
    # load data (please load data like that and let every processing step happen **inside** the train_predict function)
    # (change path if necessary)
    # X_all = np.genfromtxt("X_train.csv", delimiter=",", skip_header=1) 
    # Y_all = np.genfromtxt("y_train.csv", delimiter=",", skip_header=1) this  loading can be used instead od pandas
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
   



    # please put everything that you want to execute outside the function here!


