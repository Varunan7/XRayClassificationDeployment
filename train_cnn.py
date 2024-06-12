import os
import numpy as np
from PIL import Image

# Define paths
base_path = r'E:\Deep learning\CovidXRayImages\Val'
covid_path = os.path.join(base_path, 'COVID-19', 'images')
non_covid_path = os.path.join(base_path, 'Non-COVID', 'images')
normal_path = os.path.join(base_path, 'Normal', 'images')

# Load images and assign labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('L')
            img = np.array(img.resize((128, 128)), dtype=np.float32) / 255.0  # Reduced image size
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

covid_images, covid_labels = load_images_from_folder(covid_path, 2)
non_covid_images, non_covid_labels = load_images_from_folder(non_covid_path, 1)
normal_images, normal_labels = load_images_from_folder(normal_path, 0)

# Combine the data
images = np.array(covid_images + non_covid_images + normal_images)
labels = np.array(covid_labels + non_covid_labels + normal_labels)

# Ensure that each class has at least some samples
print(f"No infection samples: {np.sum(labels == 0)}")
print(f"Other infection samples: {np.sum(labels == 1)}")
print(f"Covid-19 samples: {np.sum(labels == 2)}")

from sklearn.utils import resample

# Combine images and labels for resampling
data = np.c_[images.reshape(len(images), -1), labels]

# Separate classes
class_0 = data[data[:, -1] == 0]
class_1 = data[data[:, -1] == 1]
class_2 = data[data[:, -1] == 2]

# Check the sizes of each class
print(f"Class 0 size: {len(class_0)}")
print(f"Class 1 size: {len(class_1)}")
print(f"Class 2 size: {len(class_2)}")

# Determine the smallest class size greater than zero
min_class_size = min(len(class_0), len(class_1), len(class_2))

# Ensure that the minimum class size is greater than zero
if min_class_size == 0:
    raise ValueError("One of the classes has zero samples. Please check the criteria and data.")

# Downsample all classes to the size of the smallest class
class_0_downsampled = resample(class_0, replace=False, n_samples=min_class_size, random_state=42)
class_1_downsampled = resample(class_1, replace=False, n_samples=min_class_size, random_state=42)
class_2_downsampled = resample(class_2, replace=False, n_samples=min_class_size, random_state=42)

# Combine downsampled classes
balanced_data = np.vstack([class_0_downsampled, class_1_downsampled, class_2_downsampled])

# Separate images and labels
images_balanced = balanced_data[:, :-1].reshape(-1, 128, 128)
labels_balanced = balanced_data[:, -1].astype(np.int)

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images_balanced, labels_balanced, test_size=0.2, random_state=42, stratify=labels_balanced)

# Reshape images to include channel dimension
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=3)
y_val = to_categorical(y_val, num_classes=3)


# Define the enhanced CNN model
def create_improved_cnn_model(input_shape):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # First Convolutional Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second Convolutional Block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third Convolutional Block
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fourth Convolutional Block
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fifth Convolutional Block
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))  # Increased size of dense layer
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # Output layer with softmax for multi-class classification

    return model


# Compile the model
def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    return model


# Create and compile the improved CNN model
input_shape = (128, 128, 1)
model = create_improved_cnn_model(input_shape)
model = compile_model(model)

# Define callbacks
callbacks = [
    EarlyStopping(patience=15, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=7, min_lr=1e-6, verbose=1)
]
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the generator to the training data
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# Train the model
model.fit(train_generator, epochs=100, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
val_auc = roc_auc_score(y_val, y_val_pred, multi_class='ovr')
val_conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

print(f'Validation AUC: {val_auc}')
print('Confusion Matrix:')
print(val_conf_matrix)

# Display confusion matrix
ConfusionMatrixDisplay(confusion_matrix=val_conf_matrix).plot()
plt.show()

# Calculate and display classification report
print('Classification Report:')
print(classification_report(y_val.argmax(axis=1), y_val_pred.argmax(axis=1)))

