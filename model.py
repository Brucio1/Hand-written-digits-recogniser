import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#Normalization
X_train = X_train/255

#Reshaped for Conv2D, it needs 4 dimensions
X_train = X_train.reshape(60000, 28, 28, 1)

#Data augmentation
datagen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   fill_mode='nearest' )
datagen.fit(X_train)

#Cancel training when accuracy reaches %99.9
class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.999):
                print("\n99.9% training accuracy reached!")
                self.model.stop_training = True

callback = myCallback()

model = tf.keras.Sequential([Conv2D(96, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             Conv2D(96, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             MaxPooling2D(2,2),

                             Conv2D(192, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             Conv2D(256, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             MaxPooling2D(2,2),

                             Conv2D(256, (3,3), activation='relu', padding='same'),
                             BatchNormalization(),
                             Conv2D(256, (3,3), activation='relu', padding='same'),

                             Flatten(),
                             Dense(256, activation='relu'),
                             BatchNormalization(),
                             Dropout(0.25),
                             Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

#The very same model achieves 99.635% test accuracy on Kaggle.
model.fit(datagen.flow(X_train, Y_train),epochs=200, callbacks=[callback])

model.save('MNIST_model.h5')

