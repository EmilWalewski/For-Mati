import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path('flower_images')

batch_size = 5
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# normalization_layer = layers.Rescaling(1./255)
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# # Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)


def build_model(hp):
    # model = Sequential([
    #     layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(10, activation='softmax')
    # ])
    cnn = Sequential()

    cnn.add(data_augmentation)
    cnn.add(Conv2D(hp.Int("input_units", min_value=180, max_value=256, step=32), (3, 3), input_shape=(img_width, img_height, 3)))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('n_layers', 1, 4)):
        cnn.add(Conv2D(hp.Int(f"conv_{i}_units", min_value=32, max_value=256, step=32), (3, 3)))
        cnn.add(Activation('relu'))

    cnn.add(Flatten())
    cnn.add(Dense(10))
    cnn.add(Activation('softmax'))

    cnn.compile(optimizer='adam',
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])

    return cnn


from keras_tuner.tuners import RandomSearch

tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=3, directory='hyper_model', project_name='flower')

from tensorflow.keras.datasets import fashion_mnist

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

tuner.search(train_ds, epochs=15, validation_data=val_ds)

print(tuner.get_best_hyperparameters()[0].values)

import pickle

with open("flowers.pkl", "wb") as f:
    pickle.dump(tuner, f)

# model = build_model()
#
# model.summary()
#
# epochs = 10
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=epochs
# )
#
# model.save("flower_2", save_format='h5')
#
# data_dir = pathlib.Path('test')
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)
#
# print(acc)
# print(val_acc)
# print(loss)
# print(val_loss)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
