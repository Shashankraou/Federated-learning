import flwr as fl
import sys
import numpy as np
import tensorflow as tf
from model import create_model
from dataset import load_client_data
from sklearn.metrics import precision_score, recall_score, f1_score

client_id = int(sys.argv[1])
print(f"FedProx Client {client_id} started...")

model = create_model()
(train_x, train_y), (test_x, test_y) = load_client_data(client_id)

# FedProx hyperparameter
MU = 0.1  # Proximal term coefficient (same as server)

class FedProxClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        # Store global model weights for proximal term
        global_weights = parameters
        model.set_weights(parameters)

        print(f"FedProx Client {client_id}: Training started")

        # Custom training loop with proximal term
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        # Convert to TensorFlow constants for efficiency
        global_weights_tf = [tf.constant(w) for w in global_weights]

        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        dataset = dataset.shuffle(len(train_x)).batch(32)

        # Training loop (2 epochs like FedAvg)
        for epoch in range(2):
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    # Standard cross-entropy loss
                    predictions = model(x_batch, training=True)
                    ce_loss = loss_fn(y_batch, predictions)

                    # Proximal term: μ/2 * ||w - w_global||²
                    proximal_term = 0.0
                    for w, w_global in zip(model.trainable_weights, global_weights_tf):
                        proximal_term += tf.reduce_sum(tf.square(w - w_global))
                    
                    proximal_loss = (MU / 2.0) * proximal_term

                    # Total FedProx loss
                    total_loss = ce_loss + proximal_loss

                # Update weights
                gradients = tape.gradient(total_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print(f"FedProx Client {client_id}: Training finished")
        return model.get_weights(), len(train_x), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        loss, acc = model.evaluate(test_x, test_y, verbose=0)
        y_pred = np.argmax(model.predict(test_x, verbose=0), axis=1)

        precision = precision_score(test_y, y_pred, average="macro", zero_division=0)
        recall = recall_score(test_y, y_pred, average="macro", zero_division=0)
        f1 = f1_score(test_y, y_pred, average="macro", zero_division=0)

        return loss, len(test_x), {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FedProxClient(),
)