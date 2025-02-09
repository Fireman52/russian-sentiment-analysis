import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List


def train_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        num_epochs: int,
        device: str = 'cuda'
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train the model and return training history.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        optimizer: Optimizer for training
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        Tuple containing training losses, training accuracies and test accuracies
    """
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')

        model.train()
        total_loss = 0
        correct_predictions = 0

        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions.double() / len(train_loader.dataset)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy.item())

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        # Evaluation
        model.eval()
        test_correct_predictions = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                test_correct_predictions += torch.sum(preds == labels)

        test_accuracy = test_correct_predictions.double() / len(test_loader.dataset)
        test_accuracies.append(test_accuracy.item())
        print(f'Test Accuracy: {test_accuracy:.4f}')

    return train_losses, train_accuracies, test_accuracies
