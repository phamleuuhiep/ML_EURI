import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_network.data.preprocess import BATCH_SIZE
from neural_network.models.cnn_model import MyCNN
from neural_network.data.preprocess import test_dataset

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    epoch_loss = 0.0
    epoch_correct = 0
    with torch.no_grad():
        model.eval()
        for (data_, target_) in dataloader:
            target_ = target_.type(torch.LongTensor)
            data_, target_ = data_.to(device), target_.to(device)

            outputs = model(data_)

            loss = loss_fn(outputs, target_)
            epoch_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            epoch_correct += torch.sum(pred == target_).item()
    return epoch_correct / size, epoch_loss / num_batches

def main():
    # Initialize your model, dataloaders, etc.
    model = MyCNN(num_classes=3)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_acc, val_loss = test(test_loader, model, criterion, device)
    print(f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}')

if __name__ == "__main__":
    main()