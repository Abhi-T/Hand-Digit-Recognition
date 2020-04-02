#importing the keras libs
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#step 1 building CNN
#initializing the CNN
classifier=Sequential()

#first convolutional layer and pooling
classifier.add(Convolution2D(32, (3,3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#second convolutional layer n pooling
classifier.add(Convolution2D(32, (3,3), activation='relu'))
#input_shape is going to be the pooled feature maps from the previous convolutional layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening the layers
classifier.add(Flatten())

#Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=6, activation='softmax')) # softmax for more than 2 classification

#compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #categorical_crossentropy for more than 2 class

#step2 prepairing the test/train data and training model
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('data/train',
                                               target_size=(64,64),
                                               batch_size=5,
                                               color_mode='grayscale',
                                               class_mode='categorical')

test_set=train_datagen.flow_from_directory('data/test',
                                               target_size=(64,64),
                                               batch_size=5,
                                               color_mode='grayscale',
                                               class_mode='categorical')

classifier.fit_generator(training_set, steps_per_epoch=500, #no of images in training set
                            epochs=10,validation_data=test_set, validation_steps=30 )# no of images in test set

#saving the model
model_json=classifier.to_json()
with open('model-bw.json','w') as json_file:
    json_file.write(model_json)
classifier.save_weights('model_bw.h5')

