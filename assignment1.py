import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

import tensorflow.keras as kb
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

train_folder = "train"
validation_folder = "test"

#data pipeline for handling preprocessing and augmentation
def data_processing():
    # load the images and create
    train = ImageDataGenerator(rescale=1/255, shear_range=0.2).flow_from_directory(train_folder,
                                                                  target_size=(32,32),                                                             
                                                                  batch_size=32,
                                                                  class_mode='binary')

    test = ImageDataGenerator(rescale=1/255).flow_from_directory(validation_folder,
                                                                    target_size=(32,32),                                                             
                                                                    batch_size=32,
                                                                    class_mode='binary')
    return train, test

# process to define the model
def construct_model():
  cnn = kb.Sequential()

  # this is the first block (conv layer + max pooling + batch norm + dropout)
  cnn.add(Conv2D(64, (2,2), activation = "relu", padding = "same", input_shape=(32, 32,3)))
  cnn.add(Conv2D(64, (2,2), activation = "relu", padding = "same"))
  cnn.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
  cnn.add(BatchNormalization())
  cnn.add(Dropout(0.4))

  # second block but now with more filters 
  cnn.add(Conv2D(128, (2,2), activation = "relu", padding = "same"))
  cnn.add(Conv2D(128, (2,2), activation = "relu", padding = "same"))
  cnn.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
  cnn.add(BatchNormalization())
  cnn.add(Dropout(0.4))

  # third block with even MORE filters 
  cnn.add(Conv2D(256, (2,2), activation = "relu", padding = "same"))
  cnn.add(Conv2D(256, (2,2), activation = "relu", padding = "same"))
  cnn.add(Conv2D(256, (2,2), activation = "relu", padding = "same"))
  cnn.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
  cnn.add(BatchNormalization())
  cnn.add(Dropout(0.3))

  # fourth block with even even more filters 
  cnn.add(Conv2D(512, (2,2), activation = "relu", padding = "same"))
  cnn.add(Conv2D(512, (2,2), activation = "relu", padding = "same"))
  cnn.add(Conv2D(512, (2,2), activation = "relu", padding = "same"))
  cnn.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
  cnn.add(BatchNormalization())
  cnn.add(Dropout(0.2))

  # flatten and ffnn portion
  cnn.add(Flatten())
  cnn.add(Dense(4000, activation='relu'))
  cnn.add(Dense(1500, activation='relu'))
  cnn.add(Dense(1, activation='sigmoid')) #predict our 2 categories

  cnn.summary()

  return cnn

# pipeline to compile and fit the model (training, validation, and predictions)
def compile_and_fit(model, train, test, ep, batch_size=32):
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy', kb.metrics.Precision(), kb.metrics.Recall()])
    history = model.fit(train, epochs=ep, batch_size=batch_size, validation_data=test)
    model.save('CIFAKE_cnn.h5')

    #Evaluaion of the model
    model.evaluate(test)

    return history



'''two functions to plot the loss and accuracy of the models'''
def plot_loss(res):
  plt.figure(figsize=(10, 10))
  plt.plot(res.history["loss"], label = "train loss")
  plt.plot(res.history["val_loss"], label = "validation loss")
  plt.title("Categorical Cross Entropy per Epoch")
  plt.xlabel("Epochs passed")
  plt.ylabel("Categorical Cross Entropy")
  plt.legend()
  plt.savefig('CNN_images/lossPerEpoch350.png')
  plt.show()



def plot_accuracy(res):
  plt.figure(figsize=(10, 10))
  plt.plot(res.history["accuracy"], label = "train accuracy")
  plt.plot(res.history["val_accuracy"], label = "validation accuracy")
  plt.title("Accuracy per Epoch")
  plt.xlabel("Epochs passed")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.savefig('CNN_images/accuracyPerEpoch350.png')
  plt.show()



''' plotting the results of our predicions of what our model predicts vs true class '''
def plot_predictions(cnn, test):
    # get some test images

    cnn = kb.models.load_model('CIFAKE_cnn.h5')
    
    images, labels = next(test)
    images = images / np.max(images) * 255

    # make predictions
    predictions = cnn.predict(test)

    # visualize predictions
    _, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        # plot image
        ax.imshow(images[i].astype('uint8'))
        ax.axis('off')

        # set title to predicted class and true class
        pred_class = 1 if predictions[i] > 0.5 else 0
        true_class = labels[i]
        ax.set_title(f"Pred: {pred_class}, True: {true_class}",
                    color=("green" if pred_class == true_class else "red"))

    plt.savefig("CNN_images/predVStrue350.png")



def plot_confusion_matrix(model, test):
    # create and plot confusion matrix
    predictions = model.predict(test)
    y_pred = (predictions > 0.6).astype(int)  # Thresholding at 0.6
    confusion_matrix = metrics.confusion_matrix(test.classes, y_pred)
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()
    plt.xlabel(f"Accuracy: {accuracy:.2f}")
    plt.show()
    plt.savefig("CNN_images/cnn_confusion_matrix350.png")
    

# entire pipeline from data processing and augmentation to training, validation, and predictions
if __name__ == "__main__":
    train, test = data_processing()
    cnn = construct_model()
    history = compile_and_fit(cnn, train, test, ep=350, batch_size=32)
    plot_loss(history)
    plot_accuracy(history)
    plot_predictions(cnn, test)
    plot_confusion_matrix(cnn, test)

'''
useful links 

https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50
https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b

'''
