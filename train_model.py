import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    'dataset_split/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_gen = datagen.flow_from_directory(
    'dataset_split/val',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

base_model = Xception(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=10)

model.save('xception_model.h5')
print("✅ Model training complete and saved as xception_model.h5!")
