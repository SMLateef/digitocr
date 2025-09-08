# model.py
from tensorflow.keras import layers, models

def build_baseline_cnn(input_shape=(28,28,1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='baseline_cnn')
    return model

# Optional: skeleton CRNN for line/sequence OCR (advanced)
def build_crnn(input_shape=(32, 128, 1), num_classes=80):  # example input HxW
    # Note: training CRNN requires CTC loss setup and sequence labels; this is a starting template.
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128,(3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # collapse height dimension
    shape = x.shape
    x = layers.Permute((2,1,3))(x)  # W, H, C
    t = layers.TimeDistributed(layers.Flatten())(x)  # W time steps
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(t)
    x = layers.Dense(num_classes+1, activation='softmax')(x)  # +1 for CTC blank
    model = models.Model(inputs, x, name='crnn')
    return model
