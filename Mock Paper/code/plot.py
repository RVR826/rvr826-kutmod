from benchmark import *
import matplotlib.pyplot as plt

def runtime_vs_size(data):
    mean_times = []
    for run in data:
        mean_times.append(np.mean(run))

    plt.plot(np.array(SIZES), mean_times, marker='o')
    plt.title("LU Decomposition Runtime vs Matrix Size")
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Mean Runtime (seconds)")
    plt.grid(True)
    plt.show()

def distribution_vs_size(data):
    times_data = {}
    for i in range(len(SIZES)):
        times_data[SIZES[i]] = data[i]

    plt.boxplot(times_data.values(), labels=times_data.keys())
    plt.title(f"Runtime Distribution over {REPEAT} Runs")
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.show()

def log_log(data):
    mean_times = []
    for run in data:
        mean_times.append(np.mean(run))

    plt.loglog(SIZES, mean_times, 'o-', base=10)
    plt.title("Complexity Scaling of LU Decomposition")
    plt.xlabel("Matrix Size (log scale)")
    plt.ylabel("Runtime (log scale)")
    plt.grid(True, which="both")
    plt.show()

def res_error(data):
    fig, axes = plt.subplots(1, len(SIZES), figsize=(15, 4))
    fig.suptitle("Residual Error Distribution per Matrix Size", fontsize=14)

    for i in range(len(SIZES)):
        ax = axes[i]
        ax.hist(data[i], bins=15, edgecolor='black')
        ax.set_title(f"n = {SIZES[i]}")
        ax.set_xlabel("Residual Error (‖PA − LU‖ₙ)")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.show()

if __name__ == "__main__":
    data = errors()
    res_error(data)