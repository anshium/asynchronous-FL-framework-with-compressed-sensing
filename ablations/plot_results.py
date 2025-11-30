import matplotlib.pyplot as plt
import json
import os
import glob

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

def load_metrics(experiment_name_pattern):
    files = glob.glob(os.path.join(RESULTS_DIR, f"metrics_{experiment_name_pattern}*.json"))
    data = {}
    for f in files:
        name = os.path.basename(f).replace("metrics_", "").replace(".json", "")
        with open(f, "r") as fp:
            data[name] = json.load(fp)
    return data

def plot_metric(data, metric, title, filename, xlabel="Epoch"):
    plt.figure(figsize=(10, 6))
    for name, points in data.items():
        x = [p['epoch'] for p in points]
        y = [p[metric] for p in points]
        plt.plot(x, y, label=name, marker='o')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()
    print(f"Saved plot: {filename}")

def main():
    # 1. Method Comparison
    data = load_metrics("method_comparison")
    if data:
        plot_metric(data, "accuracy", "Method Comparison: Accuracy", "method_accuracy.png")
        plot_metric(data, "loss", "Method Comparison: Loss", "method_loss.png")

    # 2. Compression Impact
    data = load_metrics("compression_impact")
    if data:
        plot_metric(data, "accuracy", "Compression Ratio Impact: Accuracy", "compression_accuracy.png")

    # 3. Sparsity Sensitivity
    data = load_metrics("sparsity_sensitivity")
    if data:
        plot_metric(data, "accuracy", "Sparsity Threshold Sensitivity: Accuracy", "sparsity_accuracy.png")

    # 4. Scalability
    data = load_metrics("scalability")
    if data:
        plot_metric(data, "accuracy", "Scalability: Accuracy", "scalability_accuracy.png")

if __name__ == "__main__":
    main()
