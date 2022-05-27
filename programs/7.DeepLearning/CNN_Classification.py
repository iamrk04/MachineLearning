# Please unzip dataset.zip first from https://drive.google.com/file/d/1jyTU7XFESDlx8pqus5xsF1iBrSLUetMa/view?usp=sharing

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# 1 Data Preprocessing
# 1.1 Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,        # rescale does feature scaling
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('7.DeepLearning/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')     # class_mode can be binary or categorical

# 1.2 Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('7.DeepLearning/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# 2 Building the CNN
# 2.1 Initialising the CNN
cnn = tf.keras.models.Sequential()

# 2.2  Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))      # 3 in input shape is because we are using colored image (RGB), for black and white use 1 

# 2.3 Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2.4 Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2.5 Flattening
cnn.add(tf.keras.layers.Flatten())

# 2.6 Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# 2.7 Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# 3 Training the CNN
# 3.1 Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 3.2 Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# 4 Making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('7.DeepLearning/dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
print("Class Indices \t=>t", training_set.class_indices)
print("Result \t\t=>", result)
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
print("Prediction \t=>", prediction)