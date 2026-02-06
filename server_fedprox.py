import flwr as fl
import csv
import os
from typing import Optional, Tuple, Dict

CSV_FILE = "fedprox_metrics.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "accuracy", "precision", "recall", "f1_score"])

def weighted_average(metrics):
    total = sum(num for num, _ in metrics)
    if total == 0:
        return {}

    return {
        "accuracy": sum(num * m.get("accuracy", 0) for num, m in metrics) / total,
        "precision": sum(num * m.get("precision", 0) for num, m in metrics) / total,
        "recall": sum(num * m.get("recall", 0) for num, m in metrics) / total,
        "f1_score": sum(num * m.get("f1_score", 0) for num, m in metrics) / total,
    }

class LoggingFedProx(fl.server.strategy.FedProx):
    def aggregate_evaluate(
        self,
        rnd: int,
        results,
        failures
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        aggregated = super().aggregate_evaluate(rnd, results, failures)
        if aggregated is None:
            return None

        loss, metrics = aggregated

        if metrics:
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    rnd,
                    loss,
                    metrics.get("accuracy", 0),
                    metrics.get("precision", 0),
                    metrics.get("recall", 0),
                    metrics.get("f1_score", 0),
                ])

        return loss, metrics


strategy = LoggingFedProx(
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
    evaluate_metrics_aggregation_fn=weighted_average,
    proximal_mu=0.1,   # ⭐ KEY FEDPROX PARAMETER
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
