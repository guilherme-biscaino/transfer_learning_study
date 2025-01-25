import os

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from keras import layers
from keras import Model
from pathlib import Path

root_path = Path(__file__).parent
root = root_path / 'flowers/'
batch_size = 32
img_width = 224
img_height = 224

# new data handling using tensorflow method folowing 'load and pre-process' tutorial
train_ds = tf.keras.utils.image_dataset_from_directory(
    root,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_width, img_height),
    batch_size=batch_size
    )
val_ds = tf.keras.utils.image_dataset_from_directory(
    root,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_width, img_height),
    batch_size=batch_size
    )

class_names = train_ds.class_names

num_classes = len(class_names)


# data normalization
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# using tensorflow buffers for optimal performance
AUTOTUNE = tf.data.AUTOTUNE
prepared_train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
prepared_val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# downloading vgg16 model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)

# freezing weights
base_model.trainable = False

# adding new layers to enable training
input = base_model.input
new_layer = layers.Dense(num_classes, activation="softmax")
output = new_layer(base_model.layers[-2].output)
model_new = Model(input, output)

# checking if model weights already exists otherwise train model again
if os.path.exists('flowers.weights.h5'):
    model_new.load_weights('flowers.weights.h5')

    model_new.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
else:
    model_new.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    history2 = model_new.fit(
        prepared_train_ds,
        batch_size=32,
        epochs=10,
        validation_data=prepared_val_ds)

    model_new.save_weights('flowers.weights.h5')

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(121)
    ax.plot(history2.history["val_loss"])
    ax.set_title("validation loss")
    ax.set_xlabel("epochs")

    plt.show()

# manipulation needed to ease confusion matrix creation
y_true = []
y_pred = []
for x, y in prepared_val_ds:

    y_true.append(y)
    y_pred.append(tf.argmax(model_new.predict(x), axis=1))

y_pred = tf.concat(y_pred, axis=0)
y_true = tf.concat(y_true, axis=0)

# creating and printing metrics
confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
confusion_matrix_display.plot()

plt.show()

model_metrics = []
for i, label in enumerate(class_names):
    precision, recall, fscore, support = precision_recall_fscore_support(np.array(y_true) == i, np.array(y_pred) == i)
    model_metrics.append([label, recall[0], recall[1], precision[1], fscore[1], support[1]])

df = pd.DataFrame(model_metrics, columns=["label", "especificidade",  "sensibilidade", "precisão", "f_score", "suporte"])

print("=" * 50)
print("Metricas de cada classe")
print(df)
print("=" * 50)

weight = df["suporte"]/df["suporte"].sum()
df_medium = df[["especificidade",  "sensibilidade", "precisão", "f_score"]].apply(lambda col: np.sum(col*weight, axis=0))
print("="*50)
print("Metrica das médias entre as classes")
print(df_medium)
print("="*50)
