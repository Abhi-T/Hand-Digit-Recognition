# Hand-Digit-Recognition
This is a code which detects Hand digits like 0 ,1...5

#All the data generated using webcam.
#unzip the data folder if you want to retrain or see the images.

1>Collect_Data.py contains code to capture images using webcam. Bases on input string like 0,1,2,3 etc, image will be saves in directory folder.
2>Preprocessing-Images.py contains preprocessing code, like resizing and converting to gray scale
3>train.py will train and generate the model using keras lib. It will also save the model.
4>Predict.py will load the trained keras model and predict the hand digit image using webcam.
