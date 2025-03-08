import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from neural_network.models.cnn_model import MyCNN
from neural_network.data.preprocess import train_dataset
from neural_network.data.preprocess import BATCH_SIZE

def train(dataloader, model, loss_fn, optimizer, lr_scheduler, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    for (data_, target_) in dataloader:
        target_ = target_.type(torch.LongTensor)
        data_, target_ = data_.to(device), target_.to(device)

        optimizer.zero_grad()

        outputs = model(data_)

        loss = loss_fn(outputs, target_)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        epoch_correct += torch.sum(pred == target_).item()
    lr_scheduler.step()
    return epoch_correct / size, epoch_loss / num_batches
def main():
    model = MyCNN(num_classes=3)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    lr_milestones = [7, 14, 21, 28, 35]
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    EPOCHS = 20
    for epoch in tqdm(range(EPOCHS)):
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, lr_scheduler, device)
        print(f'EPOCH: {epoch} train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f}')

if __name__ == "__main__":
    main()