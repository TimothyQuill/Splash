import matplotlib.pyplot as plt


def plot_loss(train, test, save_path="loss.jpg"):
    """
    Plot training and testing loss curves and save the figure.

    Parameters:
    train (list or array-like): Training loss values.
    test (list or array-like): Testing loss values.
    save_path (str): Path to save the plot image. Defaults to 'loss.jpg'.
    """
    plt.figure(figsize=(8, 6))  # Optional: Set figure size for better clarity
    plt.plot(train, label="Train Loss", color='blue', linestyle='-')
    plt.plot(test, label="Test Loss", color='orange', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend(loc="best")
    plt.grid(True)  # Optional: Add grid for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
