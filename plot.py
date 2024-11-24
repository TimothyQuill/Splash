import matplotlib.pyplot as plt


def plot_loss(train, test, save_path="loss.jpg"):
    """
    Plot training and testing loss curves and save the figure.

    Args:
        train (list or array-like): Training loss values.
        test (list or array-like): Testing loss values.
        save_path (str): Path to save the plot image. Defaults to 'loss.jpg'.
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train[1:], label="Train Loss", color="blue", linestyle="-")
    plt.plot(test[1:], label="Test Loss", color="orange", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
