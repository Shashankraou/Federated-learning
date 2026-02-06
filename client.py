import flwr as fl
import sys
import numpy as np
import os
from model import create_model
from dataset import load_client_data
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

client_id = int(sys.argv[1])

model = create_model()
(train_x, train_y), (test_x, test_y) = load_client_data(client_id)

class CIFARClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_x, train_y, epochs=2, batch_size=32, verbose=0)
        return model.get_weights(), len(train_x), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        loss, acc = model.evaluate(test_x, test_y, verbose=0)
        y_pred = np.argmax(model.predict(test_x, verbose=0), axis=1)

        precision = precision_score(test_y, y_pred, average="macro", zero_division=0)
        recall = recall_score(test_y, y_pred, average="macro", zero_division=0)
        f1 = f1_score(test_y, y_pred, average="macro", zero_division=0)

        # Save confusion matrix ONLY at last round
        if config.get("round", 0) == config.get("num_rounds", -1):
            os.makedirs("confusion_matrices", exist_ok=True)
            cm = confusion_matrix(test_y, y_pred)
            np.save(f"confusion_matrices/client_{client_id}.npy", cm)

        return loss, len(test_x), {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=CIFARClient(),
)
