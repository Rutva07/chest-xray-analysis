import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from diagnosis import build_model, compile_model, fgsm_attack, pgd_attack

# =======================
# 1. Load ChestMNIST Data
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
# 2. Generate Adversarial Training Data
# =======================
print("Generating adversarial training data...")
X_adv_fgsm_train = fgsm_attack(model=None, images=X_train[:15000], labels=y_train[:15000], epsilon=0.01)
X_adv_pgd_train  = pgd_attack(model=None, images=X_train[15000:30000], labels=y_train[15000:30000], epsilon=0.03, alpha=0.005, iterations=10)

X_train_combined = np.concatenate([X_train[15000:60000], X_adv_fgsm_train, X_adv_pgd_train], axis=0)
y_train_combined = np.concatenate([y_train[15000:60000], y_train[:15000], y_train[:15000]], axis=0)

perm = np.random.permutation(X_train_combined.shape[0])
X_train_combined = X_train_combined[perm]
y_train_combined = y_train_combined[perm]

# Validation adversarial data
X_adv_fgsm_val = fgsm_attack(model=None, images=X_val[:5000], labels=y_val[:5000], epsilon=0.01)
X_adv_pgd_val  = pgd_attack(model=None, images=X_val[:2000], labels=y_val[:2000], epsilon=0.03, alpha=0.005, iterations=10)
X_val_combined = np.concatenate([X_val, X_adv_fgsm_val, X_adv_pgd_val], axis=0)
y_val_combined = np.concatenate([y_val, y_val[:5000], y_val[:2000]], axis=0)

perm = np.random.permutation(X_val_combined.shape[0])
X_val_combined = X_val_combined[perm]
y_val_combined = y_val_combined[perm]

# =======================
# 3. Train Adversarially Robust Model
# =======================
adv_model = build_model(input_shape=X_train.shape[1:], num_classes=num_classes)
adv_model = compile_model(adv_model)

early_stop_adv = callbacks.EarlyStopping(
    monitor='val_auc',
    patience=5,
    mode='max',
    restore_best_weights=True
)

history_adv = adv_model.fit(
    X_train_combined, y_train_combined,
    validation_data=(X_val_combined, y_val_combined),
    epochs=20,
    batch_size=64,
    callbacks=[early_stop_adv],
    verbose=2
)

# =======================
# 4. Evaluate Robust Model
# =======================
print("Evaluating on clean test data...")
loss, acc, auc = adv_model.evaluate(X_test, y_test, verbose=2)
print(f"Clean Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")

print("Evaluating on FGSM adversarial test data...")
X_test_adv_fgsm = fgsm_attack(adv_model, X_test, y_test, epsilon=0.01)
loss, acc, auc = adv_model.evaluate(X_test_adv_fgsm, y_test, verbose=2)
print(f"FGSM Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")

print("Evaluating on PGD adversarial test data...")
X_test_adv_pgd = pgd_attack(adv_model, X_test, y_test, epsilon=0.03, alpha=0.005, iterations=10)
loss, acc, auc = adv_model.evaluate(X_test_adv_pgd, y_test, verbose=2)
print(f"PGD Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")
