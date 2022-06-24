import tensorflow as tf
from tensorflow import keras
import pathlib
import numpy as np
import os

# data_dir = pathlib.Path('guns')
# print(data_dir)


batch_size = 5
img_height = 180
img_width = 180

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
# class_names = train_ds.class_names

model = keras.models.load_model('flower_1')

# img = tf.keras.utils.load_img(
#     'test/machine_gun/mach2.jpg', target_size=(180, 180)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

folder_names = []
[folder_names.append(name) for name in os.listdir("test") if os.path.isdir('test')]
counter = 0
accuracy_counter = 0
for i in folder_names:
    root = 'test/' + i
    print("Test class: {}".format(i))
    for j in os.listdir(root):
        path = root + '/' + j

        img = tf.keras.utils.load_img(
            path, target_size=(180, 180)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        accuracy_counter = accuracy_counter + int(folder_names[np.argmax(score)] == i)
        counter = counter + 1
        print(
            "1. This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(folder_names[np.argmax(score)], 100 * np.max(score))
        )
    print('\n')
print('Accuracy counter: '+str(accuracy_counter))
print('Counter: '+str(counter))
print('Accuracy: '+str(round((accuracy_counter/counter)*100, 2)))


# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# index = 0
#
# print(
#     "1. This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
#
#
# max = 0
# for idx, val in enumerate(score):
#     if val > max:
#         max = val
#         index = idx
#
# classes = []
# for idx, val in enumerate(class_names):
#     if idx != np.argmax(score):
#         classes.append(val)
#
# score = np.delete(score.numpy(), index, axis=None)
# score = tf.constant(score)
#
# print(
#     "2. This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(classes[np.argmax(score)], 100 * np.max(score))
# )
#
# max = 0
# for idx, val in enumerate(score):
#     if val > max:
#         max = val
#         index = idx
# classes2 = []
# for idx, val in enumerate(classes):
#     if idx != np.argmax(score):
#         classes2.append(val)
# score = np.delete(score.numpy(), index, axis=None)
# score = tf.constant(score)
#
#
# print(
#     "3. This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(classes2[np.argmax(score)], 100 * np.max(score))
# )