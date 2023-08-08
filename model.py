# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:24:44 2023

@author: Adrien M.
"""

import  cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint

mnist = tf.keras.datasets.mnist # Import the dataset of 60.000 real handwritten digits.



# CREATING THE MODEL
(x_train, y_train), (x_test, y_test) = mnist.load_data()    # x = digit image, y = actual number of the digit.
x_train = x_train.astype(np.float32)/255    # normalize the grayscale value of the pixels.
x_test = x_test.astype(np.float32)/255
x_train = np.expand_dims(x_train, -1)   # Add a dimension for the channel (grayscale).
x_test = np.expand_dims(x_test, -1)    
y_train = tf.keras.utils.to_categorical(y_train)    # Convert labels to one-hot encoding.
y_test = tf.keras.utils.to_categorical(y_test)    

model = tf.keras.models.Sequential()    # Creation of the neural network.
model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'))    # Add a 2D convolutional layer with 32 filters of size 3x3, ReLU activation.
model.add(tf.keras.layers.MaxPool2D((2,2)))     # Add a max pooling layer with pool size 2x2.
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))     # Add another 2D convolutional layer with 64 filters of size 3x3, ReLU activation.
model.add(tf.keras.layers.MaxPool2D((2,2)))     # Add another max pooling layer with pool size 2x2.
model.add(tf.keras.layers.Flatten())    # Flatten the data for the fully connected layers.
model.add(tf.keras.layers.Dropout(0.25))    # Apply dropout regularization with a rate of 0.25.
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))    # Add a fully connected layer with 10 output units and softmax activation for classification.

model.compile(optimizer = 'adam', loss = tf.keras.losses.categorical_crossentropy, metrics = ['accuracy'])  # Compile the model.

es = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 4, verbose = 1)   # Stop the training if some conditions are not satisfied.
mc = ModelCheckpoint("./bestmodel.h5", monitor="val_accuracy", verbose=1, save_best_only=True)  # Save the model.

cb = [es, mc] 



# TRAINING
history = model.fit(x_train, y_train, epochs = 20, validation_split = 0.3, callbacks = cb)



# TESTING (MODEL EVALUATION)
best_model = tf.keras.models.load_model("bestmodel.h5")
performance = best_model.evaluate(x_test, y_test)
print(f"loss: {performance[0]}")
print(f"accuracy: {performance[1]}")



# PREDICTING DRAWN DIGITS (WITH PAINT)
true_digits = [6, 2, 4, 8, 7, 3, 0, 5, 1, 5, 7, 8, 2, 4, 0, 9, 1, 3, 9, 8]
prediction_success_rate = 0

for image_number in range(1, 21):
    img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]      # Read the image file and extract only the first channel (grayscale).
    img = np.invert(np.array([img]))    # Invert the pixel values (background to white, digit to black) and convert to a NumPy array.
    img = tf.keras.utils.normalize(img, axis=1)     # Normalize the pixel values to a range between 0 and 1.

    prediction = model.predict(img)     # Use the trained model to predict the digit in the image.

    print(f"The digit number {image_number} is probably: {np.argmax(prediction)}")      # Print the predicted digit label.
    plt.imshow(img[0], cmap=plt.cm.binary)      # Display the normalized image using binary colormap (black and white).
    plt.show()   

        
    if (np.argmax(prediction) == true_digits[image_number - 1]):
        prediction_success_rate += 1
        print("GOOD GUESS! \n")
    else:
        print("WRONG GUESS!")
        print(f"The digit number {image_number} is a {true_digits[image_number - 1]} and was said to be a {np.argmax(prediction)} \n")
        
prediction_success_rate = (prediction_success_rate / 20)*100
print(f"Success rate of the predictions: {prediction_success_rate}%.")