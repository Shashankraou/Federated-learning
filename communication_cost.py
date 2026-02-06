from model import create_model

model = create_model()
model_size_bytes = sum(w.numpy().nbytes for w in model.weights)
model_size_mb = model_size_bytes / (1024 * 1024)

NUM_ROUNDS = 5
NUM_CLIENTS = 5

print(f"Model size per client: {model_size_mb:.2f} MB")
print(f"Total communication cost: {model_size_mb * NUM_ROUNDS * NUM_CLIENTS:.2f} MB")
