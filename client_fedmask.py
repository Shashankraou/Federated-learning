import flwr as fl
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

from model import create_model
from dataset import load_client_data

# -----------------------------
# Client ID
# -----------------------------
client_id = int(sys.argv[1])
print(f"FedMask Client {client_id} started and waiting for server...")

# -----------------------------
# Load model and data
# -----------------------------
model = create_model()
(train_x, train_y), (test_x, test_y) = load_client_data(client_id)

# -----------------------------
# Create binary mask with ACTUAL sparsity
# -----------------------------
def create_mask(weights, sparsity=0.5):
    """
    Creates a binary mask for each weight tensor.
    sparsity: fraction of weights to zero out (0.5 = 50% sparse)
    """
    mask = []
    for w in weights:
        # Create binary mask: 1 with probability (1-sparsity), 0 with probability sparsity
        m = np.random.binomial(1, 1 - sparsity, size=w.shape)
        mask.append(m.astype(np.float32))
    return mask

# ⭐ CORRECTED: Enable 50% sparsity for true FedMask behavior
SPARSITY = 0.5
mask = create_mask(model.get_weights(), sparsity=SPARSITY)

print(f"FedMask Client {client_id}: Mask created with {SPARSITY*100}% sparsity")

def apply_mask(weights, mask):
    """Apply binary mask to weights (element-wise multiplication)"""
    return [w * m for w, m in zip(weights, mask)]

# -----------------------------
# FedMask Client
# -----------------------------
class FedMaskClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        # Send masked weights to server
        return apply_mask(model.get_weights(), mask)

    def fit(self, parameters, config):
        # Receive global masked weights and apply local mask
        model.set_weights(apply_mask(parameters, mask))

        print(f"FedMask Client {client_id}: Training started")
        
        # Data Augmentation using tf.keras.Sequential
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomRotation(0.1),  # Added rotation
        ])

        # Create tf.data pipeline
        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_ds = train_ds.shuffle(len(train_x)).batch(32)
        # Apply augmentation on the fly
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y), 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        
        # Train with more epochs for masked networks
        model.fit(train_ds, epochs=5, verbose=0)
        
        print(f"FedMask Client {client_id}: Training finished")

        # Return masked weights
        return apply_mask(model.get_weights(), mask), len(train_x), {}

    def evaluate(self, parameters, config):
        # Apply mask before evaluation
        model.set_weights(apply_mask(parameters, mask))

        loss, acc = model.evaluate(test_x, test_y, verbose=0)
        y_pred = model.predict(test_x, verbose=0).argmax(axis=1)

        precision = precision_score(test_y, y_pred, average="macro", zero_division=0)
        recall = recall_score(test_y, y_pred, average="macro", zero_division=0)
        f1 = f1_score(test_y, y_pred, average="macro", zero_division=0)

        return loss, len(test_x), {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

# -----------------------------
# Start client
# -----------------------------
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FedMaskClient(),
)