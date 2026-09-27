import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

def build_model(input_shape=(256, 256, 3)):
    """Builds and returns the Cat vs Dog CNN Sequential architecture."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),

        Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),

        Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()