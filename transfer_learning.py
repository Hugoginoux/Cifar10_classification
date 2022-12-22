import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


from keras.applications import VGG19
from keras.utils import to_categorical

vgg = VGG19(include_top=False, weights='imagenet', input_shape=(32,32,3), classes=test_labels.shape[1])
vgg.trainable = False

model_transferred = models.Sequential()
model_transferred.add(vgg)
model_transferred.add(layers.Flatten())

model_transferred.add(layers.Dense(256, activation=('relu'), input_dim=512))
model_transferred.add(layers.Dense(128, activation='relu'))
model_transferred.add(layers.Dense(64, activation='relu'))
model_transferred.add(layers.Dense(32, activation='relu'))
model_transferred.add(layers.Dense(10))

model_transferred.summary()

# train
model_transferred.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

history = model_transferred.fit(
    train_images, train_labels, epochs=35, 
    validation_data=(test_images, test_labels),
    batch_size=128
)

model_transferred.save('model_transferred')