import matplotlib.pyplot as plt

def plot_metrics(logs):
    """
    Plots training and validation loss and accuracy.

    Args:
        logs (dict): A dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(logs['train_loss'], label='Train Loss', color='blue', linestyle='-', marker='o')
    plt.plot(logs['val_loss'], label='Validation Loss', color='red', linestyle='--', marker='s')
    plt.title('Train & Validation Loss', fontsize=18, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(logs['train_acc'], label='Train Accuracy', color='green', linestyle='-', marker='o')
    plt.plot(logs['val_acc'], label='Validation Accuracy', color='orange', linestyle='--', marker='s')
    plt.title('Train & Validation Accuracy', fontsize=18, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()