import matplotlib.pyplot as plt
import numpy as np

# 数据
clustering_data = {
    "Dataset": ["Titanic", "House Price", "Milk Quality", "Bank Churn", "Anaemia", "Market Sales"],
    "Rand-PRE": [0.5976, 0.6451, 0.6450, 0.3248, 0.5936, 0.6241],
    "Min-PRE": [0.5976, 0.6436, 0.6450, 0.3248, 0.5936, 0.6241],
    "Learn2clean": [0.6243, 0.9738, 0.7844, 0.3269, 0.6298, 0.6312],
    "GA-PRE": [0.9510, 0.9649, 0.8451, 0.3249, 0.6140, 0.6304],
}

classification_data = {
    "Dataset": ["Titanic", "Milk Quality", "Bank Churn", "Anaemia"],
    "Rand-PRE": [0.6949, 0.5062, 0.7884, 0.9231],
    "Min-PRE": [0.6951, 0.6868, 0.7884, 0.9103],
    "Learn2clean": [0.6173, 0.7094, 0.8271, 0.9466],
    "GA-PRE": [0.8207, 0.8151, 0.8399, 0.9706],
}

regression_data = {
    "Dataset": ["Titanic", "Worker Productivity", "Milk Quality", "Bank Churn", "Anaemia", "Market Sales"],
    "Rand-PRE": [164.0172, 0.0244, 1.7486, 0.1441, 5.9918, 93.9355],
    "Min-PRE": [1163.4348, 0.0281, 1.7726, 0.1537, 5.3463, 103.9256],
    "Learn2clean": [141.3527, 0.0232, 0.7017, 0.1317, 3.5435, 73.5603],
    "GA-PRE": [57.2878, 0.0175, 0.0113, 0.1300, 3.1690, 8.5063],
}

def plot_grouped_bar(data, title, ylabel, log_scale=False):
    datasets = data["Dataset"]
    methods = ["Rand-PRE", "Min-PRE", "Learn2clean", "GA-PRE"]
    x = np.arange(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        offset = (i - 1.5) * width
        values = data[method]
        ax.bar(x + offset, values, width, label=method)

    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30)
    ax.legend()
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

# Visualization
plot_grouped_bar(clustering_data, "Clustering Experimental Results", "Score", log_scale=True)
plot_grouped_bar(classification_data, "Classification Experimental Results", "Score", log_scale=True)
plot_grouped_bar(regression_data, "Regression Experimental Results", "Error/Score", log_scale=True)