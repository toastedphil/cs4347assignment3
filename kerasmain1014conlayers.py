from __future__ import print_function
from configs import train_path101 as train_path
from configs import test_path101 as test_path
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import os

tensorboard_callback = keras.callbacks.TensorBoard(log_dir='food101runs')

batch_size = 10
num_classes = 10
epochs = 10
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'archive')
model_name = 'kerasfood1014conlayers.h5'
model_path = os.path.join(save_dir, model_name)


""""
DATA GENERATOR START FROM HERE
"""

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=100,
    class_mode='categorical',
    shuffle=True,
    seed = 1)

test_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(128, 128),
        batch_size=100,
        class_mode='categorical')

""""
DATA GENERATOR ENDS HERE
"""


""""
MODEL BUILDING START FROM HERE
"""
input_shape = (128, 128, 3)

cnn4 = Sequential()
cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn4.add(BatchNormalization())

cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Flatten())

cnn4.add(Dense(512, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(128, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(10, activation='softmax'))


""""
MODEL BUILDING ENDS HERE
"""

# initiate RMSprop optimizer
#opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

if os.path.exists(model_path):
    print("LOADING OLD MODEL")
    model.load_weights(model_path)

cnn4.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])



cnn4.fit_generator(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    callbacks=[tensorboard_callback])


cnn4.save(model_path)
