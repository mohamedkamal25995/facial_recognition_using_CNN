from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os


train_data_dir ='/kaggle/input/facial-expression-dataset-image-folders-fer2013/data/train/'
validation_data_dir ='/kaggle/input/facial-expression-dataset-image-folders-fer2013/data/val/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,    
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=512,
    class_mode='categorical',
    shuffle=True)


validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=512,
    class_mode='categorical',
    shuffle=True)

class_labels = ["angry", "disgust", "fear",
                "happy", "sad", "surprised", "neutral"]
img, label = train_generator.__next__()
# we use sequential because we move the input data one by one
model = Sequential()
# this is the input layer because we have one input in the time from multiple inputs
# we use (48,48,1) because the input size is 48*48 and we are working on grayscale which hase only 0/1
model.add(Conv2D(32, kernel_size=(3, 3),
          activation='relu', input_shape=(48, 48, 1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
# this makes all layers in a linear form
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
# this is the output layer and we have 7 outputs
# we use softmax cause this is categorical with 7 classes not sigmoid because it's used for binary or if you have 2 classes
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# now we train the model using the data set

# setting path
train_path = "/kaggle/input/facial-expression-dataset-image-folders-fer2013/data/train/"
test_path = "/kaggle/input/facial-expression-dataset-image-folders-fer2013/data/test/"

num_train_imgs = 0
for roots, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)

num_test_imgs = 0
for roots, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

print(num_train_imgs)
print(num_test_imgs)

# now train the model
epochs = 300


history = model.fit(train_generator,
                    steps_per_epoch=num_train_imgs//512,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs//512
                    )
model.save('/kaggle/working/model_file.h5')
