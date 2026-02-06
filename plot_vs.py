import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load all CSV files
# -----------------------------
fedavg = pd.read_csv("federated_metrics.csv")
fedprox = pd.read_csv("fedprox_metrics.csv")
fedper = pd.read_csv("fedper_metrics.csv")
fedmask = pd.read_csv("fedmask_metrics.csv")

# -----------------------------
# Plot Accuracy vs Rounds
# -----------------------------
plt.figure(figsize=(9, 6))

plt.plot(fedavg["round"], fedavg["accuracy"], marker="o", label="FedAvg")
plt.plot(fedprox["round"], fedprox["accuracy"], marker="o", label="FedProx")
plt.plot(fedper["round"], fedper["accuracy"], marker="o", label="FedPer")
plt.plot(fedmask["round"], fedmask["accuracy"], marker="o", label="FedMask")

plt.xlabel("Federated Rounds")
plt.ylabel("Accuracy")
plt.title("FedAvg vs FedProx vs FedPer vs FedMask")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
