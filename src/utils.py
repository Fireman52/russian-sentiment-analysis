import matplotlib.pyplot as plt
from typing import List


def plot_training_history(
        train_losses: List[float],
        train_accuracies: List[float],
        test_accuracies: List[float]
) -> None:
    """
    Plot training history.

    Args:
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        test_accuracies: List of test accuracies
    """
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies,
             label='Train Accuracy', marker='o')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies,
             label='Test Accuracy', marker='o')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
