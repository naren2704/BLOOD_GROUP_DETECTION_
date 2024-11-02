import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the MobileNetV2 model without the top layers
mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model layers to prevent updating during training
mobilenet_base.trainable = False

# Define the model
def create_model():
    model = models.Sequential()
    model.add(mobilenet_base)  # Add the MobileNetV2 base
    model.add(layers.GlobalAveragePooling2D())  # Add Global Average Pooling
    model.add(layers.Dropout(0.5))  # Add Dropout to prevent overfitting
    model.add(layers.Dense(128, activation='relu'))  # Fully connected dense layer
    model.add(layers.Dense(8, activation='softmax'))  # Output layer for classification (8 classes for blood groups)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Data augmentation to increase the variety of images and prevent overfitting
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Directory containing the dataset (ensure this path is correct)
dataset_directory = r'C:\Users\SEC\Desktop\archive\dataset_blood_group\organized'

# Training data generator
train_generator = datagen.flow_from_directory(
    dataset_directory,
    target_size=(128, 128),
    color_mode='rgb',  # MobileNetV2 expects RGB input
    class_mode='sparse',
    batch_size=32,
    shuffle=True
)

# Validation data generator (without augmentation)
validation_generator = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
    dataset_directory,
    target_size=(128, 128),
    color_mode='rgb',
    class_mode='sparse',
    batch_size=32
)

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save('fingerprint_blood_group_model.h5')

# Plot accuracy and loss over epochs
def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()

plot_metrics(history)

# Evaluate the model on validation set
val_steps = validation_generator.samples // validation_generator.batch_size
val_loss, val_accuracy = model.evaluate(validation_generator, steps=val_steps)

# Print validation accuracy as a percentage
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Make predictions
y_pred = model.predict(validation_generator)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
y_true = validation_generator.classes

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, f1_score
mse = mean_squared_error(y_true, y_pred_classes)
mae = mean_absolute_error(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

# Display metrics
print(f"Validation Loss: {val_loss}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
