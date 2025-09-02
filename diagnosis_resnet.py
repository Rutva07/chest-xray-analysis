import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

# =======================
# 1. Load ChestMNIST metadata only
# =======================
data = np.load("chestmnist_128.npz", allow_pickle=True)
X_train, y_train = data["train_images"], data["train_labels"]
X_val, y_val     = data["val_images"], data["val_labels"]
X_test, y_test   = data["test_images"], data["test_labels"]

print(data.files)
print("Training data shape:", X_train.shape, y_train.shape)

# =======================
# 2. Preprocessing function for ResNet
# =======================
def preprocess_grayscale_to_resnet(image, label):
    image = tf.image.resize(image, (224, 224))        # resize
    image = tf.image.grayscale_to_rgb(image)          # convert to 3 channels
    image = preprocess_input(image)                   # ResNet preprocessing
    return image, label

# =======================
# 3. Dataset generator (lazy loading)
# =======================
def dataset_generator(X, y):
    for i in range(len(X)):
        img = X[i].astype("float32") / 255.0         # normalize
        img = np.expand_dims(img, -1)                # grayscale channel
        yield img, y[i]

BATCH_SIZE = 8  # small batch for CPU

train_ds = tf.data.Dataset.from_generator(
    lambda: dataset_generator(X_train, y_train),
    output_signature=(
        tf.TensorSpec(shape=(128,128,1), dtype=tf.float32),
        tf.TensorSpec(shape=(y_train.shape[1],), dtype=tf.float32)
    )
)
train_ds = train_ds.map(preprocess_grayscale_to_resnet, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: dataset_generator(X_val, y_val),
    output_signature=(
        tf.TensorSpec(shape=(128,128,1), dtype=tf.float32),
        tf.TensorSpec(shape=(y_val.shape[1],), dtype=tf.float32)
    )
)
val_ds = val_ds.map(preprocess_grayscale_to_resnet, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_generator(
    lambda: dataset_generator(X_test, y_test),
    output_signature=(
        tf.TensorSpec(shape=(128,128,1), dtype=tf.float32),
        tf.TensorSpec(shape=(y_test.shape[1],), dtype=tf.float32)
    )
)
test_ds = test_ds.map(preprocess_grayscale_to_resnet, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =======================
# 4. Build ResNet50 model
# =======================
num_classes = y_train.shape[1]  # 14 labels

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = True  # freeze base model

inputs = layers.Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='sigmoid')(x)  # multi-label

model = models.Model(inputs, outputs)

# Compile with multi-label metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(multi_label=True)]
)

# =======================
# 5. Train model
# =======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# =======================
# 6. Evaluate on test set
# =======================
test_loss, test_binary_acc, test_auc = model.evaluate(test_ds)
print(f"Test Binary Accuracy: {test_binary_acc:.4f}, Test AUC: {test_auc:.4f}")
