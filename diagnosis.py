import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt

# =======================
# 1. Define Adversarial Attacks
# =======================
def fgsm_attack(model, images, labels, epsilon=0.01):
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images)
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)
    gradient = tape.gradient(loss, images)
    adv_images = images + epsilon * tf.sign(gradient)
    adv_images = tf.clip_by_value(adv_images, 0, 1)
    return adv_images

def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.005, iterations=10):
    adv_images = tf.identity(images)
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            predictions = model(adv_images)
            loss = tf.keras.losses.binary_crossentropy(labels, predictions)
        gradient = tape.gradient(loss, adv_images)
        adv_images = adv_images + alpha * tf.sign(gradient)
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0, 1)
    return adv_images

# =======================
# 2. Load ChestMNIST Data
# =======================
data = np.load("chestmnist_128.npz")
X_train, y_train = data["train_images"], data["train_labels"]
X_val, y_val     = data["val_images"], data["val_labels"]
X_test, y_test   = data["test_images"], data["test_labels"]

# Normalize and add channel
X_train = np.expand_dims(X_train.astype("float32") / 255.0, -1)
X_val   = np.expand_dims(X_val.astype("float32") / 255.0, -1)
X_test  = np.expand_dims(X_test.astype("float32") / 255.0, -1)

num_classes = y_train.shape[1]

# =======================
# 3. Build CNN Model
# =======================
def build_model(input_shape=(28,28,1), num_classes=14):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    return model

def compile_model(model):
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(multi_label=True)]
    )
    return model

# =======================
# 4. Train on Clean Data
# =======================
model = build_model(input_shape=X_train.shape[1:], num_classes=num_classes)
model = compile_model(model)

early_stop = callbacks.EarlyStopping(
    monitor='val_auc',
    patience=5,
    mode='max',
    restore_best_weights=True
)

# Sample image and adversarial examples
idx = 2
image = X_train[idx:idx+1]
label = y_train[idx:idx+1]

adv_image_fgsm = fgsm_attack(model, image, label, epsilon=0.01)
adv_image_pgd  = pgd_attack(model, image, label, epsilon=0.03, alpha=0.005, iterations=10)

plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
plt.imshow(image[0].squeeze(), cmap="gray")
plt.title(f"Original\nLabels: {label[0]}")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(adv_image_fgsm[0].numpy().squeeze(), cmap="gray")
plt.title("FGSM Attack")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(adv_image_pgd[0].numpy().squeeze(), cmap="gray")
plt.title("PGD Attack")
plt.axis("off")

plt.tight_layout()
plt.show()

# Train clean model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)

# =======================
# 5. Evaluate on Clean + Adversarial Test Data
# =======================
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=2)
print(f"Clean Model - Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

X_adv_fgsm_test = fgsm_attack(model, X_test[:1000], y_test[:1000], epsilon=0.01)
loss, acc, auc = model.evaluate(X_adv_fgsm_test, y_test[:1000], verbose=2)
print(f"FGSM Attack: Accuracy={acc:.4f}, AUC={auc:.4f}")

X_adv_pgd_test = pgd_attack(model, X_test[:1000], y_test[:1000], epsilon=0.03, alpha=0.005, iterations=10)
loss, acc, auc = model.evaluate(X_adv_pgd_test, y_test[:1000], verbose=2)
print(f"PGD Attack: Accuracy={acc:.4f}, AUC={auc:.4f}")
