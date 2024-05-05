import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
import time

start_time = time.time()

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("GPU(s) available: "),
print(tf.config.list_physical_devices('GPU'))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train_small = x_train[:1000]
y_train_small = y_train[:1000]
x_test_small = x_test[:200]
y_test_small = y_test[:200]

with tf.device('/GPU:0'):  # Change this num to switch GPU
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train_small, y_train_small, epochs=10, validation_data=(x_test_small, y_test_small))

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")
