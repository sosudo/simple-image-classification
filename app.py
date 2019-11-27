from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import numpy as np
import matplotlib.pyplot as plt
import pathlib
#print(tf.__version__)
splits = tfds.Split.ALL.subsplit(weighted=(80, 10, 10))
splits, info = tfds.load('fashion_mnist', with_info=True, as_supervised=True, split=splits)
(train_examples, validation_examples, test_examples) = splits
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
with open('labels.txt', 'w') as f:
	f.write('\n'.join(class_names))
IMG_SIZE = 28, 28
def format_example(image, label):
	image = tf.image.resize(image, IMG_SIZE) / 225.0
	return image, label
BATCH_SIZE = 32
train_batches = train_examples.shuffle(num_examples // 4).map(format_example).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_example).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(format_example).batch(1)
#model = tf.keras.Sequential([
#	tf.keras.layers.Conv2D(kernel_size=3, filters=16, activation='relu', input_shape=(28, 28, 1),
#	tf.keras.layers.MaxPooling2D(),
#	tf.keras.layers.Conv2D(kernel_size=3, filters=32, activation='relu'),
#	tf.keras.layers.Flatten(),
#	tf.keras.layers.Dense(units=64, activation='relu'),
#	tf.keras.layers.Dense(units=64, activation='softmax')
#])
model = tf.keras.Sequential([
  # Set the input shape to (28, 28, 1), kernel size=3, filters=16 and use ReLU activation,  
  tf.keras.layers.Conv2D(kernel_size=3, filters=16, activation='relu', input_shape=(28,28,1)),
  # model.add(Conv2D (kernel_size = (20,30), filters = 400, activation='relu'))    
  tf.keras.layers.MaxPooling2D(),
  # Set the number of filters to 32, kernel size to 3 and use ReLU activation 
  tf.keras.layers.Conv2D(kernel_size=3, filters=32, activation='relu'),
  # Flatten the output layer to 1 dimension
  tf.keras.layers.Flatten(),
  # Add a fully connected layer with 64 hidden units and ReLU activation
  tf.keras.layers.Dense(units=64, activation='relu'),
  # Attach a final softmax classification head
  tf.keras.layers.Dense(units=64, activation='softmax')
  # model.add(Dense(64,activation='softmax'))
])
model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)
model.fit(train_batches, epochs=10, validation_data=validation_batches)
export_dir = 'saved_model/1'
tf.saved_model.save(model, export_dir)
optimization = tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
#optimization
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_model_file = 'model.tflite'
with open(tflite_model_file, "wb") as f:
	f.write(tflite_model)
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
predictions = []
test_labels = []
test_images = []
for img, label in test_batches.take(50):
  interpreter.set_tensor(input_index, img)
  interpreter.invoke()
  predictions.append(interpreter.get_tensor(output_index))
  test_labels.append(label[0])
  test_images.append(np.array(img))
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  img = np.squeeze(img)
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label.numpy():
    color = 'green'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks(list(range(10)), class_names, rotation='vertical')
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array[0], color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array[0])
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')
index = int(input("Image Number [1-50] - "))
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(index, predictions, test_labels, test_images)
plt.show()
again = str(input("Would you like to see a different number (y/n)? "))
if again == "y":
	index = int(input("Image Number [1-50] - "))
	plt.figure(figsize=(6,3))
	plt.subplot(1,2,1)
	plot_image(index, predictions, test_labels, test_images)
	plt.show()
if again == "n":
	exit()

