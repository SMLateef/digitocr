# train_mnist.py
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from model import build_baseline_cnn

def load_prepared_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test

def train(epochs=12, batch_size=128):
    x_train, y_train, x_test, y_test = load_prepared_mnist()
    model = build_baseline_cnn(input_shape=(28,28,1), num_classes=10)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.05,
        zoom_range=0.08
    )
    datagen.fit(x_train)

    ckpt = ModelCheckpoint('best_mnist.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              validation_data=(x_test, y_test),
              steps_per_epoch=len(x_train)//batch_size,
              epochs=epochs,
              callbacks=[ckpt, es, rl])

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {acc:.4f}")
    model.save('final_mnist.h5')
    return model

if __name__ == "__main__":
    train(epochs=15)
