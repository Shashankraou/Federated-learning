import flwr as fl
import sys
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from model import create_model
from dataset import load_client_data

client_id = int(sys.argv[1])
print(f"FedPer Client {client_id} started and waiting for server...")

model = create_model()
(train_x, train_y), (test_x, test_y) = load_client_data(client_id)

# FedPer: Share first layers, personalize last layers
# Count total weight arrays
total_weights = len(model.get_weights())
SHARED_LAYERS_END = total_weights - 4  # Personalize last 2 dense layers (4 arrays: 2 kernels + 2 biases)

print(f"FedPer Client {client_id}: Total weights = {total_weights}")
print(f"FedPer Client {client_id}: Sharing first {SHARED_LAYERS_END}, personalizing last {total_weights - SHARED_LAYERS_END}")

class FedPerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        all_weights = model.get_weights()
        shared_weights = all_weights[:SHARED_LAYERS_END]
        return shared_weights

    def fit(self, parameters, config):
        current_weights = model.get_weights()
        current_weights[:SHARED_LAYERS_END] = parameters
        model.set_weights(current_weights)

        print(f"FedPer Client {client_id}: Training started")
        
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomRotation(0.1),
        ])

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_ds = train_ds.shuffle(len(train_x)).batch(32)
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        
        # Stage 1: Train entire model
        model.fit(train_ds, epochs=3, verbose=0)
        
        # Stage 2: Fine-tune personalized layers
        # Freeze shared layers
        num_layers_to_freeze = SHARED_LAYERS_END // 2
        for i in range(num_layers_to_freeze):
            if hasattr(model.layers[i], 'trainable'):
                model.layers[i].trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        model.fit(train_ds, epochs=2, verbose=0)
        
        # Unfreeze for next round
        for layer in model.layers:
            if hasattr(layer, 'trainable'):
                layer.trainable = True
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        print(f"FedPer Client {client_id}: Training finished")

        trained_weights = model.get_weights()
        shared_weights = trained_weights[:SHARED_LAYERS_END]
        return shared_weights, len(train_x), {}

    def evaluate(self, parameters, config):
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

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FedPerClient(),
)