import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import time

def train_model():
    # 檢查設備是否可用
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 定義資料轉換
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 加載 CIFAR-10 數據集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)

    # 使用預設的 ResNet-18 模型
    model = models.resnet18(weights=None, num_classes=10)
    model = model.to(device)

    # 定義損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練模型並計時
    total_time = 0
    epochs = 10  # 訓練 10 個 epochs

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time += epoch_time

        print(f"Epoch {epoch+1}/{epochs} - 損失值: {total_loss/len(train_loader):.4f} - 耗時: {epoch_time:.2f} 秒")

    print(f"全部 {epochs} 個 epochs 訓練完成，總耗時: {total_time:.2f} 秒")

# 解決 Windows 多進程問題
if __name__ == '__main__':
    train_model()
