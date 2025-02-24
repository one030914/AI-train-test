import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import time

def train_model():
    # 使用 XPU 設備
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"使用設備: {device}")

    # CIFAR-10 資料預處理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 載入 CIFAR-10 數據集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)

    # 初始化 ResNet-50 模型
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=None, num_classes=10)
    model = model.to(device)

    # 定義損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練模型
    total_time = 0
    epochs = 10

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # 正向傳播
            output = model(data)
            loss = criterion(output, target)

            # 反向傳播與優化
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time += epoch_time

        print(f"Epoch {epoch+1}/{epochs} - 損失值: {total_loss / len(train_loader):.4f} - 耗時: {epoch_time:.2f} 秒")

    print(f"全部 {epochs} 個 epochs 訓練完成，總耗時: {total_time:.2f} 秒")


if __name__ == '__main__':
    train_model()
