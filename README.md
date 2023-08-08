# Handwritten digit recognition

Very simple machine learning project aiming at making the computer recognise handwritten digits.

The aim of the project was to build a machine learning program to train the computer for handrwitten digit recognition. So that the user can draw a digit in a window an the computer tells what digit it is.

There are 2 (two) Python files: application.py and model.py. 

## model.py:
The model.py file' s aim is to build the machine learning model. For this purpose the mnist database is used to train the model, the performances of the model are testing (accuracy over 98%) and then the model has to predict the digit drawn on paint on 20 (twenty) different images. 
For this prediction test, a good answer rate is obtained and is never so good (between 55 and 75%), maybye due to the quality of the input images. For more information, please go to the following path: 


## application.py:
The application.py file' s aim is to build the black board that allows the user to draw digits that will be recognised by the computer. Here we can see all the digits drawn on the black board and guess correctly: 

![Screenshot](screenshot/all_digits.png)

For more pictures please go to the following path: 


## Use:
If you want to use the files, please run the model.py first so that a bestmodel.h5 file will be created and used later by the application.py file. Then run the application.py file.
