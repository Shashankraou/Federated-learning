import flwr as fl
import sys
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from model import create_model
from dataset import load_client_data

# -----------------------------
# Client ID
# -----------------------------
client_id = int(sys.argv[1])
print(f"FedPer Client {client_id} started and waiting for server...")

# -----------------------------
# Load model and data
# -----------------------------
model = create_model()
(train_x, train_y), (test_x, test_y) = load_client_data(client_id)

# -----------------------------
# FedPer Configuration
# -----------------------------
# In FedPer, we split the model into:
# - Shared layers: trained collaboratively across clients
# - Personalized layers: kept local to each client

# For the current model architecture:
# Layer 0: Conv2D(32) + bias = weights[0:2]
# Layer 1: MaxPooling (no weights)
# Layer 2: Conv2D(64) + bias = weights[2:4]
# Layer 3: MaxPooling (no weights)
# Layer 4: Flatten (no weights)
# Layer 5: Dense(128) + bias = weights[4:6]
# Layer 6: Dense(10) + bias = weights[6:8]  ← Personalized layer

# So we have 8 weight arrays total (4 layers × 2 for kernel+bias)
# Share first 6 weights (Conv+Dense), personalize last 2 (final Dense)

SHARED_LAYERS_END = 6  # Share everything except the final classification layer

print(f"FedPer Client {client_id}: Sharing first {SHARED_LAYERS_END} weight arrays")
print(f"FedPer Client {client_id}: Personalizing last {len(model.get_weights()) - SHARED_LAYERS_END} weight arrays")

# -----------------------------
# Federated Personalization Client (FedPer)
# -----------------------------
class FedPerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        # 🔥 Send ONLY shared layers to server
        all_weights = model.get_weights()
        shared_weights = all_weights[:SHARED_LAYERS_END]
        return shared_weights

    def fit(self, parameters, config):
        # 🔥 Update ONLY shared layers from server
        # Keep personalized layers unchanged
        current_weights = model.get_weights()
        current_weights[:SHARED_LAYERS_END] = parameters
        model.set_weights(current_weights)

        print(f"FedPer Client {client_id}: Training started")
        
        # Two-stage training for better personalization
        
        # Stage 1: Train entire model (including personalized layer)
        model.fit(train_x, train_y, epochs=3, batch_size=32, verbose=0)
        
        # Stage 2: Fine-tune personalized layer more
        # Freeze shared layers
        for i in range(SHARED_LAYERS_END // 2):  # Freeze first N layers
            model.layers[i].trainable = False
        
        # Recompile with frozen layers
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Fine-tune personalized layers
        model.fit(train_x, train_y, epochs=2, batch_size=32, verbose=0)
        
        # Unfreeze for next round
        for layer in model.layers:
            layer.trainable = True
        
        # Recompile back to normal
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        print(f"FedPer Client {client_id}: Training finished")

        # 🔥 Return ONLY shared layers to server
        trained_weights = model.get_weights()
        shared_weights = trained_weights[:SHARED_LAYERS_END]
        return shared_weights, len(train_x), {}

    def evaluate(self, parameters, config):
        # Update shared layers from server
        current_weights = model.get_weights()
        current_weights[:SHARED_LAYERS_END] = parameters
        model.set_weights(current_weights)

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
    client=FedPerClient(),
)