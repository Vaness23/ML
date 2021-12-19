import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

digit_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digit_mnist.load_data(path="mnist.npz")

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Массивы train_images и train_labels — это данные, которые использует модель для обучения
# Массивы test_images и test_labels используются для тестирования модели

# class_names = ['Футболка', 'Штаны', 'Пулловер', 'Платье', 'Пальто', 
#                'Сандали', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки']

class_names = ['0', '1', '2', '3', '4', 
                '5', '6', '7', '8', '9']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])

# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # переформатирование данных из 2D массива (28х28 пикселей) в 1D массив (784 пикселя)
    keras.layers.Dense(128, activation=tf.nn.relu), # первый слой из 128 узлов
    keras.layers.Dense(10, activation=tf.nn.softmax)# второй слой из 10 узлов, возвращающий массив из 10 вероятностных оценок (сумма = 1)
])

model.compile(optimizer=tf.optimizers.Adam(),   # это то, как модель обновляется на основе данных, которые она видит, и функции потери
              loss='sparse_categorical_crossentropy',# измеряет насколько точная модель во время обучения
              metrics=['accuracy'])                 # используется для контроля за этапами обучения и тестирования

print("#############################")
print("Learning on Training Set")
# начинаем обучение
model.fit(train_images, train_labels, epochs=5)


print("#############################")
print("Learning on Test Set")
# обучение на тестовом наборе данных вместо тренировочного
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Loss: ", test_loss, "; Test Acc: ", test_acc)

# прогнозирование
predictions = model.predict(test_images)

print(predictions[0])

print(np.argmax(predictions[0]))

print(test_labels[0])


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

plt.show()

# # Возьмём изображение из тестового набора данных
# img = test_images[0]

# #Добавим изображение в пакет, где он является единственным членом
# img = (np.expand_dims (img, 0))

# predictions_single = model.predict(img)
# print(predictions_single)

# plot_value_array(0, predictions_single, test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)

# print(np.argmax(predictions_single[0]))

# plt.show()