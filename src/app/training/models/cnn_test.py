# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 07:13:21 2021

@author: voodo
"""
import os
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(15)
tf.config.threading.set_intra_op_parallelism_threads(15)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
#import tensorflow.keras.datasets
from sklearn.model_selection import train_test_split
#%matplotlib inline


(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

print('Training data shape: {},{}'.format(train_X.shape, train_Y.shape))
print('Testing data shape: {},{}'.format(test_X.shape, test_Y.shape))

#Using TensorFlow backend.

classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

#create our plot object
#plt.figure(figsize=[5,5])

# Display the first image in training data
#plt.subplot(121)
#plt.imshow(train_X[0,:,:], cmap='gray')
#plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
#plt.subplot(122)
#plt.imshow(test_X[0,:,:], cmap='gray')
#plt.title("Ground Truth : {}".format(test_Y[0]))


#Reshape the data by adding another dimension
train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
print("Train, test after reshape:  {},{}".format(train_X.shape, test_X.shape))


#re-cast as float and normalize between 0 and 1
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

#split the training data
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print("train: {}, validate: {}, train label: {}, validation label: {}".format(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape))


batch_size = 128 #128/256 -- RAM dependent.
epochs = 20
num_classes = 10


#Create network layers and add to the model in their sequence
test_model = Sequential()
test_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
test_model.add(LeakyReLU(alpha=0.1))
test_model.add(MaxPool2D((2, 2),padding='same'))
test_model.add(Dropout(0.25))
test_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
test_model.add(LeakyReLU(alpha=0.1))
test_model.add(MaxPool2D(pool_size=(2, 2),padding='same'))
test_model.add(Dropout(0.25))
test_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
test_model.add(LeakyReLU(alpha=0.1))                  
test_model.add(MaxPool2D(pool_size=(2, 2),padding='same'))
test_model.add(Dropout(0.4))
test_model.add(Flatten())
test_model.add(Dense(128, activation='linear'))
test_model.add(LeakyReLU(alpha=0.1))    
test_model.add(Dropout(0.3))              
test_model.add(Dense(num_classes, activation='softmax'))

#compile the model
test_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

#from keras import backend as K
#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))



#Summarize the model
test_model.summary()

#Train the model with checkpoints
if not os.path.isdir("cnnCheck"):
    os.mkdir("cnnCheck")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='cnnCheck',
                                                 save_weights_only=True,
                                                 verbose=1)

test_train = test_model.fit(train_X, 
                            train_label, 
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(valid_X, valid_label),
                            callbacks=[cp_callback])



#model evaluation on test
test_eval = test_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#Save the model with the train and test values
if not os.path.isdir("cnn"):
    os.mkdir("cnn")
test_train.save("cnn\{}".format('test_train_{}{}'.format(test_eval[0],test_eval[1])))


#plot accuracy and loss between training and validation
#TODO save off the plots and associate with the model
accuracy = test_train.history['accuracy']
val_accuracy = test_train.history['val_accuracy']
loss = test_train.history['loss']
val_loss = test_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()