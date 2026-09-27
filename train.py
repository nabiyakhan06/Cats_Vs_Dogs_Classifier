import os
import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_model

def load_data(train_dir='train', test_dir='test', batch_size=32, img_size=(256, 256)):
    """Loads and normalizes dataset batches without .cache() to prevent RAM crashes."""
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError("Please ensure 'train' and 'test' dataset folders exist in root directory.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=img_size
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=img_size
    )

    def process(image, label):
        image = tf.cast(image / 255.0, tf.float32)
        return image, label

    train_ds = train_ds.map(process)
    validation_ds = validation_ds.map(process)
    return train_ds, validation_ds

def plot_history(history):
    """Plot accuracy and loss curves."""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], color='red', label='train')
    plt.plot(history.history['val_accuracy'], color='blue', label='validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], color='red', label='train')
    plt.plot(history.history['val_loss'], color='blue', label='validation')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Loading datasets...")
    train_ds, validation_ds = load_data()
    
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nStarting model training...")
    history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
    
    # Save model in native Keras format
    model.save('cat_dog_model.keras')
    print("\nModel successfully saved to 'cat_dog_model.keras'")

    plot_history(history)