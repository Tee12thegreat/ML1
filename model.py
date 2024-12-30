import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Define the base path for your dataset
base_path = r'C:\Users\Z\Desktop\data'  # Adjust this path to match your data location

# Data Preparation
def load_data(base_path):
    classes = ['benign', 'malignant', 'normal']
    images = []
    labels = []

    for label in classes:
        class_dir = os.path.join(base_path, label)
        if not os.path.exists(class_dir):
            print(f"Directory does not exist: {class_dir}")
            continue
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = load_img(img_path, target_size=(224, 224))
                    img_array = img_to_array(img) / 255.0  # Normalize the image
                    images.append(img_array)
                    labels.append(classes.index(label))  # Use index as label
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)

# Model Architecture
def create_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of base_model
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the Model
def train_model(model, images, labels):
    model.fit(images, labels, epochs=15, batch_size=32)
    model.save('breast_cancer_model.h5')

if __name__ == '__main__':
    images, labels = load_data(base_path)
    
    if len(images) > 0 and len(labels) > 0:
        model = create_model(num_classes=len(set(labels)))
        train_model(model, images, labels)
