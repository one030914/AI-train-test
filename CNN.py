import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# 檢查 XPU 是否可用
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"使用設備: {device}")

# 定義資料轉換
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下載 MNIST 數據集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定義簡單 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 動態計算輸入特徵大小
        self._to_linear = None
        self._get_conv_output()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_conv_output(self):
        # 建立一個虛擬輸入來計算展平之前的特徵大小
        x = torch.randn(1, 1, 28, 28)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        self._to_linear = x.numel()  # 計算展平後的特徵數量

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)  # 自動計算展平形狀
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型與優化器
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 訓練模型並計時
start_time = time.time()

model.train()
for epoch in range(1):  # 只跑一輪來測試效能
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'訓練進度: [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}')

end_time = time.time()
print(f"訓練完成，花費時間: {end_time - start_time:.2f} 秒")
