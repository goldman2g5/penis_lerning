"""
Для того чтобы нейросеть работала надо читать эту молитву каждый раз:

Отче наш, иже еси в моем PC
Да святится имя и расширение Твое
Да придет прерывание Твое и да будет воля твоя
BASIC наш насущный дай нам;
И прости нам дизассемблеры и антивирусы наши,
как Копиpайты прощаем мы.
И не введи нас в Exception Error,
но избавь нас от зависания,
ибо Твое есть адpессное пространство,
порты и регистры
Во имя CTRLа, ALTa и Святого DELa,
всемогущего RESETa
во веки веков,
ENTER.

Раз прочитано: 5

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import models, layers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


path = './kaggle/'
data = pd.read_csv(path + 'icml_face_data.csv')
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
classes = dict(zip(range(0, 7), (((data[data[' Usage'] == 'Training']['emotion'].value_counts()).sort_index()) / len(
    data[data[' Usage'] == 'Training']['emotion'])).tolist()))
X = data.emotion
kf = KFold(n_splits=6)
for train, test in kf.split(X):
    print("%s %s" % (train, test))


print(data)


def parse_data(deta):
    image_array = np.zeros(shape=(len(deta), 48, 48))
    image_label = np.array(list(map(int, deta['emotion'])))

    for i, row in enumerate(deta.index):
        image = np.fromstring(deta.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


# Делим на трейн, валидацию и тест, спасибо usage за облегчение моих мучений
train_imgs, train_lbls = parse_data(data[data[" Usage"] == "Training"])
val_imgs, val_lbls = parse_data(data[data[" Usage"] == "PrivateTest"])
test_imgs, test_lbls = parse_data(data[data[" Usage"] == "PublicTest"])
train_images = train_imgs.reshape((train_imgs.shape[0], 48, 48, 1))
train_images = train_images.astype('float32') / 255
val_images = val_imgs.reshape((val_imgs.shape[0], 48, 48, 1))
val_images = val_images.astype('float32') / 255
test_images = test_imgs.reshape((test_imgs.shape[0], 48, 48, 1))
test_images = test_images.astype('float32') / 255


print("train shape", np.shape(train_imgs))
print("validation shape", np.shape(val_imgs))
print("validatio shape", np.shape(val_imgs))
print(train_imgs)


model_mlp = keras.Sequential()
model_mlp.add(layers.Flatten(input_shape=(48, 48, 1)))
model_mlp.add(layers.Dense(units=120, activation='relu'))
model_mlp.add(layers.Dense(units=84, activation='relu'))
model_mlp.add(layers.Dense(units=7, activation='softmax'))
model_mlp.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])

model_mlp.summary()


model_mlp.fit(train_imgs, train_lbls,
              epochs=10, batch_size=32,
              validation_data=(val_imgs, val_lbls), verbose=1)


train_labels = to_categorical(train_lbls)
val_labels = to_categorical(val_lbls)
test_labels = to_categorical(test_lbls)


model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='selu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='selu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='relu'))
model_cnn.add(layers.Dense(7, activation='sigmoid'))
model_cnn.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model_cnn.summary()


history = model_cnn.fit(train_images, train_labels,
                        validation_data=(val_images, val_labels),
                        class_weight=classes,
                        epochs=12,
                        batch_size=512)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


test_prob = model_cnn.predict(test_images)
test_pred = np.argmax(test_prob, axis=1)
test_accuracy = np.mean(test_pred == test_lbls)
print(test_accuracy)


conf_mat = confusion_matrix(test_lbls, test_pred)
pd.DataFrame(conf_mat, columns=emotions.values(), index=emotions.values())


"""
# Код ниже стоит под большим вопросом нужен ли он, просто хуева туча нейросетей


model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='selu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='selu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='selu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='selu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='selu'))
model_cnn.add(layers.Dense(7, activation='softmax'))


model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='selu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='selu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='selu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='selu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='selu'))
model_cnn.add(layers.Dense(7, activation='sigmoid'))


model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='elu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='elu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='elu'))
model_cnn.add(layers.Dense(7, activation='softmax'))


model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='elu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='elu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='elu'))
model_cnn.add(layers.Dense(7, activation='sigmoid'))


model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='relu'))
model_cnn.add(layers.Dense(7, activation='sigmoid'))


model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='relu'))
model_cnn.add(layers.Dense(7, activation='sigmoid'))


model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='relu'))
model_cnn.add(layers.Dense(7, activation='softmax'))


model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='elu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='selu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='relu'))
model_cnn.add(layers.Dense(7, activation='softmax'))

model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='elu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='elu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='selu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='selu'))
model_cnn.add(layers.Dense(7, activation='sigmoid'))

model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(128, (3, 3), activation='selu', input_shape=(48, 48, 1)))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='selu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='elu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='relu'))
model_cnn.add(layers.Dense(7, activation='sigmoid'))

"""
