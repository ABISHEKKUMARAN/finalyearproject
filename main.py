import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


from mobilevit import *

num_classes = 4
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to the [0, 1] range
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    horizontal_flip=True,  # Randomly flip images horizontally
    zoom_range=0.2  # Randomly zoom in on images
)

# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(256, 256),  
    batch_size=32,
    class_mode='categorical'
)

valid_datagen = ImageDataGenerator(
    rescale=1.0/255)

valid_generator = valid_datagen.flow_from_directory(
    'data/test',
    target_size=(256, 256),  
    batch_size=32,
    class_mode='categorical'
)

model = MobileViT_S(input_shape=(256, 256, 3), num_classes=4)  # Updated input_shape to (224, 224, 3)
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=valid_generator
)


model.save('model1.h5')
print('model saved')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(valid_generator)
print(f'Test accuracy: {test_accuracy}')

# Plot accuracy and loss curves
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.show()
