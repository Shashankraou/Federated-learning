import pandas as pd
import matplotlib.pyplot as plt

# Load federated metrics
df = pd.read_csv("federated_metrics.csv")

# Plot metrics
plt.figure(figsize=(8, 5))
plt.plot(df["round"], df["accuracy"], marker="o", label="Accuracy")
plt.plot(df["round"], df["f1_score"], marker="o", label="F1-score")
plt.plot(df["round"], df["precision"], marker="o", label="Precision")
plt.plot(df["round"], df["recall"], marker="o", label="Recall")

plt.xlabel("Federated Rounds")
plt.ylabel("Metric Value")
plt.title("Federated Learning Convergence")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

