import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def compute(self):
        if self.total == 0: return 0
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

# ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2) 

        self.fc = nn.Linear(256 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Overfit and Train

def fit_one_batch(model, trainloader, optimizer, criterion, device, accuracy):
    print("\n--- Task 1: Overfitting on a single batch ---")
    model.train()
    
    inputs, labels = next(iter(trainloader))
    inputs, labels = inputs.to(device), labels.to(device)
    
    for i in range(100): 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        accuracy.update(outputs, labels)
        acc = accuracy.compute()

        print(f"[{i + 1}] loss: {loss.item():.4f}, accuracy: {acc}")
        
        accuracy.reset()


def train_full(model, trainloader, testloader, optimizer, criterion, device, accuracy, epochs=15):
    print(f"\n--- Task 2: Training for {epochs} epochs ---")
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        accuracy.reset()
        
        print(f"\nEpoch {epoch + 1} start:")

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accuracy.update(outputs, labels)

            # log every 100 batches
            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                acc = accuracy.compute()
                
                print(f"[{i + 1}] loss: {avg_loss:.4f}, accuracy: {acc}")
                
                running_loss = 0.0
                accuracy.reset()
        
        scheduler.step()
        evaluate(model, testloader, device)

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f'Test accuracy: {acc * 100:.2f}%')


if __name__ == '__main__':
    
    TASK_MODE = 'OVERFIT'   # task 1
    # TASK_MODE = 'TRAIN'   # task 2

    stats = ((0.4915, 0.4822, 0.4466), (0.2463, 0.2428, 0.2607))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    accuracy = Accuracy()

    if TASK_MODE == 'OVERFIT':
        print(">>> MODE: OVERFIT")
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        fit_one_batch(model, trainloader, optimizer, criterion, device, accuracy)

    elif TASK_MODE == 'TRAIN':
        print(">>> MODE: TRAIN")
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001) 
        
        train_full(model, trainloader, testloader, optimizer, criterion, device, accuracy, epochs=15)