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
# Create binary mask
# -----------------------------
def create_mask(weights, sparsity=0.5):
    mask = []
    for w in weights:
        m = np.random.binomial(1, 1 - sparsity, size=w.shape)
        mask.append(m.astype(np.float32))
    return mask

# Masking disabled for better accuracy (standard FedAvg)
mask = create_mask(model.get_weights(), sparsity=0.0)

def apply_mask(weights, mask):
    return [w * m for w, m in zip(weights, mask)]

# -----------------------------
# FedMask Client
# -----------------------------
class FedMaskClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return apply_mask(model.get_weights(), mask)

    def fit(self, parameters, config):
        model.set_weights(apply_mask(parameters, mask))

        print(f"FedMask Client {client_id}: Training started")
        
        # Modern Data Augmentation using tf.data and Keras Layers
        # This fixes the UserWarning about PyDataset and is more performant
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_ds = train_ds.shuffle(len(train_x)).batch(32)
        # Apply augmentation on the fly on the GPU (if available)
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        model.fit(train_ds, epochs=5, verbose=0)
        print(f"FedMask Client {client_id}: Training finished")

        return apply_mask(model.get_weights(), mask), len(train_x), {}

    def evaluate(self, parameters, config):
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
