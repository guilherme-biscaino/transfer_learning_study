import os

import random
import numpy as np
import keras

import matplotlib.pyplot as plt
from keras_preprocessing import image
from tf_keras.applications.imagenet_utils import preprocess_input 
from keras import layers
from keras import Model
from pathlib import Path

root_path = Path(__file__).parent
root = root_path / 'flowers/'
train_split, val_split = 0.7, 0.15

categories = [x[0] for x in os.walk(root) if x[0]][1:]
categories = [c for c in categories if c not in [os.path.join(root)]]


def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


data = []
for c, category in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames
              in os.walk(category) for f in filenames
              if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
    for img_path in images:
        img, x = get_image(img_path)
        data.append({'x': np.array(x[0]), 'y': c})

num_classes = len(categories)

random.shuffle(data)

idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]

# normalize data
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

vgg = keras.applications.VGG16(weights='imagenet', include_top=True)

# make a reference to VGG's input layer
inp = vgg.input

# make a new softmax layer with num_classes neurons
new_classification_layer = layers.Dense(num_classes, activation='softmax')

# connect our new layer to the second to last layer in VGG, and make a reference to it
out = new_classification_layer(vgg.layers[-2].output)

# create a new network between inp and out
model_new = Model(inp, out)

if os.path.exists(root_path / 'flowers.weights.h5'):
    model_new.load_weights('flowers.weights.h5')
else:
    # make all layers untrainable by freezing weights (except for last layer)
    for l, layer in enumerate(model_new.layers[:-1]):
        layer.trainable = False

    # ensure the last layer is trainable/not frozen
    for l, layer in enumerate(model_new.layers[-1:]):
        layer.trainable = True

    model_new.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    history2 = model_new.fit(x_train, y_train,
                            batch_size=32,
                            epochs=10,
                            validation_data=(x_val, y_val))

    model_new.save_weights('flowers.weights.h5')

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(121)
    ax.plot(history2.history["val_loss"])
    ax.set_title("validation loss")
    ax.set_xlabel("epochs")

    plt.show()

target_shape = (224, 224)

def test_image(file_path, model):
    img = image.load_img(file_path, target_size=target_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)

    class_probabilities = predictions[0]

    predicted_class_index = np.argmax(class_probabilities)

    return class_probabilities, predicted_class_index


test_image_file = '/content/flowers/daisy/100080576_f52e8ee070_n.jpg' #image path for testing
class_probabilities, predicted_class_index = test_image(test_image_file, model_new)

for i, class_label in enumerate(categories):
    probability = class_probabilities[i]
    print(f'Class: {class_label}, Probability: {probability:.4f}')

predicted_class = categories[predicted_class_index]
print(f'The image is classified as: {predicted_class}')
