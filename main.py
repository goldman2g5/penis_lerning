import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix


global elite
global zalupka321


def exists(filePath):
    try:
        os.stat(filePath)
    except OSError:
        return False
    return True

tf.random.set_seed(0)
var = tf.keras.backend.clear_session

tdat = pd.read_csv('kaggle/icml_face_data.csv')
tdat.sample(5)

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

tdat.info()

dat = tdat.copy()
dat.drop_duplicates(inplace=True)
dat.info()

dat[' Usage'].unique()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
sns.countplot(data=dat[dat[' Usage'] == 'Training'], x='emotion', ax=ax1).set_title('Training')
ax1.set_xticklabels(emotions.values())
sns.countplot(data=dat[dat[' Usage'] == 'PublicTest'], x='emotion', ax=ax2).set_title('Testing')
ax2.set_xticklabels(emotions.values())
sns.countplot(data=dat[dat[' Usage'] == 'PrivateTest'], x='emotion', ax=ax3).set_title('Validation')
ax3.set_xticklabels(emotions.values())


def prepare_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        img = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        img = np.reshape(img, (48, 48))
        image_array[i] = img

    return image_array, image_label


def sample_plot(x, y=None):
    # x, y are numpy arrays
    n = 20
    samples = random.sample(range(x.shape[0]), n)

    figs, axs = plt.subplots(2, 10, figsize=(25, 5), sharex=True, sharey=True)
    ax = axs.ravel()
    for i in range(n):
        ax[i].imshow(x[samples[i], :, :], cmap=plt.get_cmap('gray'))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if y is not None:
            ax[i].set_title(emotions[y[samples[i]]])


train_image_array, train_image_label = prepare_data(dat[dat[' Usage'] == 'Training'])
val_image_array, val_image_label = prepare_data(dat[dat[' Usage'] == 'PrivateTest'])
test_image_array, test_image_label = prepare_data(dat[dat[' Usage'] == 'PublicTest'])

train_images = train_image_array.reshape((train_image_array.shape[0], 48, 48, 1))
train_images = train_images.astype('float32')
val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1))
val_images = val_images.astype('float32')
test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1))
test_images = test_images.astype('float32')

train_labels = tf.keras.utils.to_categorical(train_image_label)
val_labels = tf.keras.utils.to_categorical(val_image_label)
test_labels = tf.keras.utils.to_categorical(test_image_label)

sample_plot(val_image_array, val_image_label)

sample_plot(test_image_array, test_image_label)

if exists("kaggle/model/models.h5") is True:
    model = load_model("kaggle/model/models.h5")

else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 255, input_shape=(48, 48, 1)),
        tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2),
        tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'),

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(16, 5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(emotions), activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    earlystop = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1e-3, restore_best_weights=True)
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-7)

    wt = dat[dat[' Usage'] == "Training"].groupby('emotion').agg('count')

    wt['fraction'] = wt[' pixels'] / np.sum(wt[' pixels'])
    class_weights = dict(zip(range(7), wt.fraction))

    hist = model.fit(train_images, train_labels,
                     validation_data=(val_images, val_labels),
                     epochs=50,
                     class_weight=class_weights,
                     batch_size=128,
                     callbacks=[earlystop, lr])
    for key in hist.history.keys():
        plt.plot(hist.history[key], label=key)

    plt.legend()

    model.evaluate(test_images, test_labels)
    test_pred = model.predict(test_images)
    confusion_matrix(y_true=test_image_label, y_pred=np.argmax(test_pred, axis=1))

    model.summary()

    model.save("kaggle/model/models.h5")


prediction = model.predict(train_images)
n = 1122
zalupka321 = "".join(list(emotions[np.argmax(prediction[n])]))

print("Распознан объект: ", zalupka321)

# ХУЙНЯ
