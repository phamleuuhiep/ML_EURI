import torch
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from neural_network.models.cnn_model import MyCNN
from neural_network.data.preprocess import train_loader, test_loader
from neural_network.training.train import train
from neural_network.training.evaluate import test

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = MyCNN(num_classes=3).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer setup
    learning_rate = 0.0001
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    lr_milestones = [7, 14, 21, 28, 35]
    multi_step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    # Training parameters
    EPOCHS = 20
    patience = 5
    counter = 0
    best_loss = float('inf')

    # Logs for visualization
    logs = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
    }

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, multi_step_lr_scheduler, device)
        val_acc, val_loss = test(test_loader, model, criterion, device)

        print(f'EPOCH: {epoch} '
              f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f} '
              f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f} '
              f'Learning Rate: {optimizer.param_groups[0]["lr"]}')

        logs['train_loss'].append(train_loss)
        logs['train_acc'].append(train_acc)
        logs['val_loss'].append(val_loss)
        logs['val_acc'].append(val_acc)

        # Save model
        torch.save(model.state_dict(), "last.pth")
        if val_loss < best_loss:
            counter = 0
            best_loss = val_loss
            torch.save(model.state_dict(), "best.pth")
        else:
            counter += 1

        if counter >= patience:
            print("Early stop!")
            break

if __name__ == "__main__":
    main()