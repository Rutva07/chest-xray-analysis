import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt

# =======================
# 1. Define Adversarial Attacks
# =======================


def fgsm_attack(model, images, labels, epsilon=0.2):
    """
    Generate adversarial examples using FGSM (Fast Gradient Sign Method)
    """
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


def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, iterations=10):
    """
    Generate adversarial examples using PGD (Projected Gradient Descent)
    """
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

# Normalize to [0,1] and add channel dimension
X_train = np.expand_dims(X_train.astype("float32") / 255.0, -1)
X_val   = np.expand_dims(X_val.astype("float32") / 255.0, -1)
X_test  = np.expand_dims(X_test.astype("float32") / 255.0, -1)

num_classes = y_train.shape[1]  # 14 labels



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


# =======================
# 4. Compile Model
# =======================
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
# 5. Training on Clean Data
# =======================

model = build_model()
model = compile_model(model)

early_stop = callbacks.EarlyStopping(
    monitor='val_auc',
    patience=5,
    mode='max',
    restore_best_weights=True
)


# =============================
# Just for printing an example
# =============================

idx = 2
image = X_train[idx:idx+1]   
label = y_train[idx:idx+1]

# Generate adversarial examples
adv_image_fgsm = fgsm_attack(model, image, label, epsilon=0.01)
adv_image_pgd  = pgd_attack(model, image, label, epsilon=0.03, alpha=0.005, iterations=10)
orig_img = image[0].squeeze()
fgsm_img = adv_image_fgsm[0].numpy().squeeze()
pgd_img  = adv_image_pgd[0].numpy().squeeze()

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(orig_img, cmap="gray")
plt.title(f"Original\nLabels: {y_train[idx]}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(fgsm_img, cmap="gray")
plt.title("FGSM Attack")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pgd_img, cmap="gray")
plt.title("PGD Attack")
plt.axis("off")

plt.tight_layout()
plt.show()
print("Training on clean data...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)




# =======================
# 6. Evaluate Trained Model
# =======================
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=2)
print(f"Clean Model - Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

X_sample, y_sample = X_test[:1000], y_test[:1000] 

X_adv = fgsm_attack(model, X_sample, y_sample, epsilon=0.01) 
adv_loss, adv_acc, adv_auc = model.evaluate(X_adv, y_sample, verbose=2) 
print(f"FGSM Attack: Accuracy={adv_acc:.4f}, AUC={adv_auc:.4f}") 

X_adv_pgd = pgd_attack(model, X_sample, y_sample, epsilon=0.03, alpha=0.005, iterations=10) 
adv_loss, adv_acc, adv_auc = model.evaluate(X_adv_pgd, y_sample, verbose=2) 
print(f"PGD Attack: Accuracy={adv_acc:.4f}, AUC={adv_auc:.4f}") 


"""
Test Accuracy: 0.9480, Test AUC: 0.7450 1/1 - 0s - 45ms/step - auc: 0.1670 - binary_accuracy: 0.9688 - loss: 0.2116 FGSM Attack: Accuracy=0.9688, AUC=0.1670 1/1 - 0s - 36ms/step - auc: 0.0627 - binary_accuracy: 0.9330 - loss: 0.3788 PGD Attack: Accuracy=0.9330, AUC=0.0627
"""


# =======================
# 7. Generate Adversarial Training and Validation Data
# =======================
print("Generating adversarial training data...")
X_adv_fgsm_train = fgsm_attack(model, X_train[:15000], y_train[:15000], epsilon=0.01)
X_adv_pgd_train  = pgd_attack(model, X_train[15000:30000], y_train[15000:30000], epsilon=0.03, alpha=0.005, iterations=10)
y_adv_fgsm_train = y_train[:15000]
y_adv_pgd_train  = y_train[:15000]

X_train_combined = np.concatenate([X_train[15000:60000], X_adv_fgsm_train, X_adv_pgd_train], axis=0)
y_train_combined = np.concatenate([y_train[15000:60000], y_adv_fgsm_train, y_adv_pgd_train], axis=0)

perm = np.random.permutation(X_train_combined.shape[0])
X_train_combined = X_train_combined[perm]
y_train_combined = y_train_combined[perm]

# Validation adversarial examples
print("Generating adversarial validation data...")
X_adv_fgsm_val = fgsm_attack(model, X_val[:5000], y_val[:5000], epsilon=0.01)
X_adv_pgd_val  = pgd_attack(model, X_val[:2000], y_val[:2000], epsilon=0.03, alpha=0.005, iterations=10)
y_adv_fgsm_val = y_val[:5000]
y_adv_pgd_val  = y_val[:2000]

X_val_combined = np.concatenate([X_val, X_adv_fgsm_val, X_adv_pgd_val], axis=0)
y_val_combined = np.concatenate([y_val, y_adv_fgsm_val, y_adv_pgd_val], axis=0)

perm = np.random.permutation(X_val_combined.shape[0])
X_val_combined = X_val_combined[perm]
y_val_combined = y_val_combined[perm]


# =======================
# 8. Train Adversarially Robust Model
# =======================
adv_model = build_model()
adv_model = compile_model(adv_model)

early_stop_adv = callbacks.EarlyStopping(
    monitor='val_auc',
    patience=5,
    mode='max',
    restore_best_weights=True
)

print("Training on adversarial + clean data...")
history_adv = adv_model.fit(
    X_train_combined, y_train_combined,
    validation_data=(X_val_combined, y_val_combined),
    epochs=20,
    batch_size=64,
    callbacks=[early_stop_adv],
    verbose=2
)


# =======================
# 9. Evaluate Adversarially Trained Model
# =======================
print("Evaluating on clean test data...")
test_loss, test_acc, test_auc = adv_model.evaluate(X_test, y_test, verbose=2)
print(f"Clean Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

print("Evaluating on FGSM adversarial test data...")
X_test_adv_fgsm = fgsm_attack(adv_model, X_test, y_test, epsilon=0.01)
adv_loss, adv_acc, adv_auc = adv_model.evaluate(X_test_adv_fgsm, y_test, verbose=2)
print(f"FGSM Test Accuracy: {adv_acc:.4f}, Test AUC: {adv_auc:.4f}")

print("Evaluating on PGD adversarial test data...")
X_test_adv_pgd = pgd_attack(adv_model, X_test, y_test, epsilon=0.03, alpha=0.005, iterations=10)
adv_loss, adv_acc, adv_auc = adv_model.evaluate(X_test_adv_pgd, y_test, verbose=2)
print(f"PGD Test Accuracy: {adv_acc:.4f}, Test AUC: {adv_auc:.4f}")


"""
Epoch 1/20
2025-09-01 00:48:58.900848: E tensorflow/core/util/util.cc:131] oneDNN supports DT_BOOL only on platforms with AVX-512. Falling back to the default Eigen-based implementation if present.
1227/1227 - 39s - 32ms/step - auc: 0.5843 - binary_accuracy: 0.9440 - loss: 0.1957 - val_auc: 0.6449 - val_binary_accuracy: 0.9492 - val_loss: 0.1744
Epoch 2/20
1227/1227 - 38s - 31ms/step - auc: 0.6412 - binary_accuracy: 0.9485 - loss: 0.1755 - val_auc: 0.7101 - val_binary_accuracy: 0.9492 - val_loss: 0.1701
Epoch 3/20
1227/1227 - 41s - 33ms/step - auc: 0.6658 - binary_accuracy: 0.9485 - loss: 0.1722 - val_auc: 0.7008 - val_binary_accuracy: 0.9492 - val_loss: 0.1675
Epoch 4/20
1227/1227 - 37s - 30ms/step - auc: 0.6849 - binary_accuracy: 0.9486 - loss: 0.1699 - val_auc: 0.7251 - val_binary_accuracy: 0.9492 - val_loss: 0.1648
Epoch 5/20
1227/1227 - 37s - 30ms/step - auc: 0.6949 - binary_accuracy: 0.9486 - loss: 0.1683 - val_auc: 0.7247 - val_binary_accuracy: 0.9492 - val_loss: 0.1664
Epoch 6/20
1227/1227 - 37s - 31ms/step - auc: 0.7070 - binary_accuracy: 0.9487 - loss: 0.1670 - val_auc: 0.7318 - val_binary_accuracy: 0.9492 - val_loss: 0.1638
Epoch 7/20
1227/1227 - 38s - 31ms/step - auc: 0.7118 - binary_accuracy: 0.9487 - loss: 0.1659 - val_auc: 0.7431 - val_binary_accuracy: 0.9494 - val_loss: 0.1677
Epoch 8/20
1227/1227 - 39s - 31ms/step - auc: 0.7190 - binary_accuracy: 0.9488 - loss: 0.1650 - val_auc: 0.7448 - val_binary_accuracy: 0.9493 - val_loss: 0.1619
Epoch 9/20
1227/1227 - 38s - 31ms/step - auc: 0.7209 - binary_accuracy: 0.9488 - loss: 0.1644 - val_auc: 0.7434 - val_binary_accuracy: 0.9492 - val_loss: 0.1606
Epoch 10/20
1227/1227 - 38s - 31ms/step - auc: 0.7266 - binary_accuracy: 0.9488 - loss: 0.1637 - val_auc: 0.7386 - val_binary_accuracy: 0.9493 - val_loss: 0.1617
Epoch 11/20
1227/1227 - 38s - 31ms/step - auc: 0.7283 - binary_accuracy: 0.9488 - loss: 0.1632 - val_auc: 0.7430 - val_binary_accuracy: 0.9493 - val_loss: 0.1619
Epoch 12/20
1227/1227 - 38s - 31ms/step - auc: 0.7340 - binary_accuracy: 0.9489 - loss: 0.1628 - val_auc: 0.7326 - val_binary_accuracy: 0.9492 - val_loss: 0.1644
Epoch 13/20
1227/1227 - 38s - 31ms/step - auc: 0.7374 - binary_accuracy: 0.9489 - loss: 0.1620 - val_auc: 0.7534 - val_binary_accuracy: 0.9493 - val_loss: 0.1600
Epoch 14/20
1227/1227 - 38s - 31ms/step - auc: 0.7391 - binary_accuracy: 0.9490 - loss: 0.1616 - val_auc: 0.7530 - val_binary_accuracy: 0.9493 - val_loss: 0.1598
Epoch 15/20
1227/1227 - 39s - 32ms/step - auc: 0.7423 - binary_accuracy: 0.9489 - loss: 0.1613 - val_auc: 0.7545 - val_binary_accuracy: 0.9494 - val_loss: 0.1595
Epoch 16/20
1227/1227 - 39s - 32ms/step - auc: 0.7481 - binary_accuracy: 0.9490 - loss: 0.1608 - val_auc: 0.7513 - val_binary_accuracy: 0.9494 - val_loss: 0.1595
Epoch 17/20
1227/1227 - 40s - 32ms/step - auc: 0.7461 - binary_accuracy: 0.9490 - loss: 0.1606 - val_auc: 0.7545 - val_binary_accuracy: 0.9494 - val_loss: 0.1591
Epoch 18/20
1227/1227 - 40s - 33ms/step - auc: 0.7508 - binary_accuracy: 0.9490 - loss: 0.1602 - val_auc: 0.7522 - val_binary_accuracy: 0.9493 - val_loss: 0.1594
Epoch 19/20
1227/1227 - 39s - 32ms/step - auc: 0.7507 - binary_accuracy: 0.9490 - loss: 0.1598 - val_auc: 0.7548 - val_binary_accuracy: 0.9492 - val_loss: 0.1595
Epoch 20/20
1227/1227 - 39s - 32ms/step - auc: 0.7498 - binary_accuracy: 0.9490 - loss: 0.1598 - val_auc: 0.7546 - val_binary_accuracy: 0.9493 - val_loss: 0.1592
702/702 - 3s - 5ms/step - auc: 0.7510 - binary_accuracy: 0.9478 - loss: 0.1631
Clean Model - Test Accuracy: 0.9478, Test AUC: 0.7510
32/32 - 0s - 6ms/step - auc: 0.2920 - binary_accuracy: 0.9379 - loss: 0.2641
FGSM Attack: Accuracy=0.9379, AUC=0.2920
32/32 - 0s - 6ms/step - auc: 0.1772 - binary_accuracy: 0.9166 - loss: 0.5037
PGD Attack: Accuracy=0.9166, AUC=0.1772
"""



"""Test Accuracy: 0.9480, Test AUC: 0.7450
1/1 - 0s - 45ms/step - auc: 0.1670 - binary_accuracy: 0.9688 - loss: 0.2116
FGSM Attack: Accuracy=0.9688, AUC=0.1670
1/1 - 0s - 36ms/step - auc: 0.0627 - binary_accuracy: 0.9330 - loss: 0.3788
PGD Attack: Accuracy=0.9330, AUC=0.0627
"""



"""
Epoch 1/20
C:\Python\Lib\site-packages\keras\src\callbacks\early_stopping.py:153: UserWarning: Early stopping conditioned on metric `val_auc` which is not available. Available metrics are: auc_1,binary_accuracy,loss,val_auc_1,val_binary_accual_loss
  current = self.get_monitor_value(logs)
1172/1172 - 59s - 50ms/step - auc_1: 0.5595 - binary_accuracy: 0.9450 - loss: 0.1977 - val_auc_1: 0.5677 - val_binary_accuracy: 0.9492 - val_loss: 0.2194
Epoch 2/20
1172/1172 - 48s - 41ms/step - auc_1: 0.5925 - binary_accuracy: 0.9488 - loss: 0.1799 - val_auc_1: 0.6082 - val_binary_accuracy: 0.9492 - val_loss: 0.1946
Epoch 3/20
1172/1172 - 47s - 40ms/step - auc_1: 0.6138 - binary_accuracy: 0.9488 - loss: 0.1774 - val_auc_1: 0.6631 - val_binary_accuracy: 0.9492 - val_loss: 0.1720
Epoch 4/20
1172/1172 - 44s - 37ms/step - auc_1: 0.6291 - binary_accuracy: 0.9488 - loss: 0.1756 - val_auc_1: 0.6730 - val_binary_accuracy: 0.9492 - val_loss: 0.1725
Epoch 5/20
1172/1172 - 41s - 35ms/step - auc_1: 0.6426 - binary_accuracy: 0.9488 - loss: 0.1741 - val_auc_1: 0.6750 - val_binary_accuracy: 0.9492 - val_loss: 0.1748
Epoch 6/20
1172/1172 - 41s - 35ms/step - auc_1: 0.6559 - binary_accuracy: 0.9488 - loss: 0.1728 - val_auc_1: 0.7105 - val_binary_accuracy: 0.9492 - val_loss: 0.1674
Epoch 7/20
1172/1172 - 41s - 35ms/step - auc_1: 0.6637 - binary_accuracy: 0.9488 - loss: 0.1721 - val_auc_1: 0.7109 - val_binary_accuracy: 0.9492 - val_loss: 0.1670
Epoch 8/20
1172/1172 - 42s - 35ms/step - auc_1: 0.6710 - binary_accuracy: 0.9488 - loss: 0.1710 - val_auc_1: 0.7262 - val_binary_accuracy: 0.9494 - val_loss: 0.1662
Epoch 9/20
1172/1172 - 44s - 37ms/step - auc_1: 0.6806 - binary_accuracy: 0.9489 - loss: 0.1703 - val_auc_1: 0.7166 - val_binary_accuracy: 0.9494 - val_loss: 0.1658
Epoch 10/20
1172/1172 - 42s - 36ms/step - auc_1: 0.6830 - binary_accuracy: 0.9489 - loss: 0.1697 - val_auc_1: 0.7227 - val_binary_accuracy: 0.9494 - val_loss: 0.1636
Epoch 11/20
1172/1172 - 41s - 35ms/step - auc_1: 0.6853 - binary_accuracy: 0.9489 - loss: 0.1692 - val_auc_1: 0.7246 - val_binary_accuracy: 0.9493 - val_loss: 0.1628
Epoch 12/20
1172/1172 - 42s - 36ms/step - auc_1: 0.6908 - binary_accuracy: 0.9489 - loss: 0.1687 - val_auc_1: 0.7311 - val_binary_accuracy: 0.9495 - val_loss: 0.1636
Epoch 13/20
1172/1172 - 41s - 35ms/step - auc_1: 0.6959 - binary_accuracy: 0.9489 - loss: 0.1682 - val_auc_1: 0.7300 - val_binary_accuracy: 0.9494 - val_loss: 0.1633
Epoch 14/20
1172/1172 - 41s - 35ms/step - auc_1: 0.6972 - binary_accuracy: 0.9490 - loss: 0.1678 - val_auc_1: 0.7299 - val_binary_accuracy: 0.9494 - val_loss: 0.1623
Epoch 15/20
1172/1172 - 41s - 35ms/step - auc_1: 0.6989 - binary_accuracy: 0.9490 - loss: 0.1673 - val_auc_1: 0.7146 - val_binary_accuracy: 0.9493 - val_loss: 0.1666
Epoch 16/20
1172/1172 - 41s - 35ms/step - auc_1: 0.7019 - binary_accuracy: 0.9490 - loss: 0.1672 - val_auc_1: 0.7286 - val_binary_accuracy: 0.9495 - val_loss: 0.1626
Epoch 17/20
1172/1172 - 41s - 35ms/step - auc_1: 0.7037 - binary_accuracy: 0.9490 - loss: 0.1668 - val_auc_1: 0.7341 - val_binary_accuracy: 0.9496 - val_loss: 0.1612
Epoch 18/20
1172/1172 - 42s - 35ms/step - auc_1: 0.7062 - binary_accuracy: 0.9490 - loss: 0.1664 - val_auc_1: 0.7336 - val_binary_accuracy: 0.9494 - val_loss: 0.1613
Epoch 19/20
1172/1172 - 41s - 35ms/step - auc_1: 0.7079 - binary_accuracy: 0.9490 - loss: 0.1662 - val_auc_1: 0.7379 - val_binary_accuracy: 0.9495 - val_loss: 0.1611
Epoch 20/20
1172/1172 - 41s - 35ms/step - auc_1: 0.7118 - binary_accuracy: 0.9490 - loss: 0.1660 - val_auc_1: 0.7313 - val_binary_accuracy: 0.9493 - val_loss: 0.1619
Evaluating on clean test data...
702/702 - 4s - 5ms/step - auc_1: 0.7352 - binary_accuracy: 0.9478 - loss: 0.1652
Clean Test Accuracy: 0.9478, Test AUC: 0.7352
Evaluating on FGSM adversarial test data...
702/702 - 5s - 7ms/step - auc_1: 0.2596 - binary_accuracy: 0.9439 - loss: 0.2616
FGSM Test Accuracy: 0.9439, Test AUC: 0.2596
Evaluating on PGD adversarial test data...
702/702 - 5s - 7ms/step - auc_1: 0.1751 - binary_accuracy: 0.9183 - loss: 0.4667
PGD Test Accuracy: 0.9183, Test AUC: 0.1751
"""