import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# --- 1. Data Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical'
)

# --- 2. Load Pre-trained Model ---
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base layers

# --- 3. Add Custom Classification Head ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# --- 4. Compile Model ---
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 5. Train Model ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# --- 6. Save Model ---
model.save('10_class_model.h5')

# --- 7. Predict New Image ---
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'new_image.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0

preds = model.predict(x)
predicted_class = np.argmax(preds, axis=1)
print("Predicted class:", predicted_class[0])
