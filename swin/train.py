import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

torch.manual_seed(42)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print("training on: " + device)

train_handler = h5py.File('../rec1501614399_export.hdf5', 'r')

train=pd.DataFrame(columns=['accelerator_pedal_position',
                                       'brake_pedal_status',
                                       'steering_wheel_angle'])

train['accelerator_pedal_position'] = list(train_handler['accelerator_pedal_position'])
train['steering_wheel_angle'] = list(train_handler['steering_wheel_angle'])
train['brake_pedal_status'] = list(train_handler['brake_pedal_status'])

window_size = 10

n_heads = 8
d_model = 160 #20 * n_heads

img_repr_dim = 1024

class PolicyNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.sep = -1 * torch.ones(d_model, dtype=torch.float32,
                               device = device).unsqueeze(0)
    self.img_ffn = nn.Sequential(
        nn.Linear(img_repr_dim, d_model),
        nn.ReLU()
    )
    self.sensor_ffn = nn.Sequential(
        nn.Linear(3, d_model),
        nn.ReLU()
    )
    self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
    
    self.head = nn.Sequential(
        nn.Linear(d_model, 3),
        nn.Sigmoid()
    )


  def forward(self, imgs, sensors, img_enc=None, sens_enc=None):
    imgs = self.img_ffn(imgs)
    sensors = self.sensor_ffn(sensors)
    imgs_encoded = imgs.clone().detach()
    sensors_encoded = sensors.clone().detach()

    # segment encoding if provided
    if img_enc is not None:
      for idx in range(len(imgs)):
        imgs_encoded[idx] = imgs[idx].add(img_enc[idx])
    if sens_enc is not None:
      for idx in range(len(sensors)):
        sensors_encoded[idx] = sensors[idx].add(sens_enc[idx])

    x = torch.cat((imgs_encoded, self.sep, sensors_encoded), dim=0)
    x = self.encoder(x)
    x = self.head(x[-1])

    return x

policyNet=PolicyNet()
policyNet.to(device)
print("model defined...")

criterion = nn.L1Loss()
optimizer = optim.SGD(policyNet.parameters(), lr=0.01)

alpha = 0.2

num_epochs = 300
losses = []
train_frames = torch.load("./train_frames.pt")
print(train_frames.shape)

print("training...")

for epoch in range(num_epochs):
  #initializing queues
  img_queue = []
  sensors_queue = []
  for i in range(len(train["accelerator_pedal_position"])):

    sensors = [train['accelerator_pedal_position'][i]/100,
                  train['brake_pedal_status'][i],
                  (train['steering_wheel_angle'][i]+600)/1200]

    img = train_frames[i].squeeze(0)


    if ((len(img_queue) < window_size) or (len(sensors_queue) < window_size)):
      img_queue.append(img.tolist())
      sensors_queue.append(sensors)
      continue
    else:
      img_queue.pop(0)
      sensors_queue.pop(0)
      img_queue.append(img.tolist())
      sensors_queue.append(sensors)

    if((len(img_queue) != window_size) or (len(sensors_queue) != window_size)):
      print(len(img_queue), len(sensors_queue))
      break

    # converting queues to tensors
    img_queue_tensor = torch.tensor(img_queue, dtype=torch.float32).to(device)
    sensors_queue_tensor = torch.tensor(sensors_queue, dtype=torch.float32).to(device)

    # doing a training cycle
    output = policyNet(img_queue_tensor, sensors_queue_tensor[:-1])

    optimizer.zero_grad()

    loss = criterion(output, sensors_queue_tensor[-1])
    # outliers prevention
    loss = loss - alpha * torch.abs(sensors_queue_tensor[-1][1] - sensors_queue_tensor[-2][1])
    loss = torch.nn.functional.relu(loss)

    if(output.shape != sensors_queue_tensor[-1].shape):
      print(output.shape, sensors_queue_tensor[-1].shape)
      break

    loss.backward()
    optimizer.step()
    if(epoch == num_epochs-1):
      losses.append(loss.item())

  print(f"done epoch {epoch}.")

print("training finished")

torch.save(policyNet.state_dict(), "./weights.pt")
