import flwr as fl
import sys
import tensorflow as tf
import numpy as np

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
# Federated Personalization Client (FedPer)
# -----------------------------
class CIFARClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        # 🔥 Send ONLY shared layers (exclude last Dense layer)
        return model.get_weights()[:-2]

    def fit(self, parameters, config):
        # 🔥 Update ONLY shared layers from server
        current_weights = model.get_weights()
        current_weights[:-2] = parameters
        model.set_weights(current_weights)

        print(f"FedPer Client {client_id}: Training started")
        model.fit(train_x, train_y, epochs=2, batch_size=32, verbose=0)
        print(f"FedPer Client {client_id}: Training finished")

        # 🔥 Return ONLY shared layers
        return model.get_weights()[:-2], len(train_x), {}

    def evaluate(self, parameters, config):
        current_weights = model.get_weights()
        current_weights[:-2] = parameters
        model.set_weights(current_weights)

        loss, acc = model.evaluate(test_x, test_y, verbose=0)
        return loss, len(test_x), {"accuracy": float(acc)}

# -----------------------------
# Start client
# -----------------------------
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=CIFARClient(),
)
