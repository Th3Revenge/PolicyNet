import torch
import torch.nn as nn
import h5py
import numpy as np
from torchvision.models import swin_b
from torchvision.models import Swin_B_Weights
import pandas as pd
import torch.optim as optim
from matplotlib import pyplot as plt
import os

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")
print(f"Version: {torch.__version__}, GPU: {torch.cuda.is_available()}, NUM_GPU: {torch.cuda.device_count()}")

train = h5py.File('./rec1501614399_export.hdf5', 'r')
test = h5py.File('./rec1501612590_export.hdf5', 'r')

classifier = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
classifier.head = nn.Identity()
classifier.to(device)
classifier.eval()
print("model downloaded and intialized...")
img_transforms = Swin_B_Weights.IMAGENET1K_V1.transforms()

# Generation of image representations to be used for training and testing the model

train_frames_path = './train_frames.pt'
test_frames_path = './test_frames.pt'

if not os.path.exists(train_frames_path):
  print("generating train frames representations")
  train_frames = []
  for i in range(len(train["aps_frame"])):
    img = np.flip(train["aps_frame"][i])
    img = torch.tensor(img.copy(), dtype=torch.float32).to(device)
    img = img.unsqueeze(0)
    img = img.repeat(3, 1, 1)
    img = img_transforms(img.unsqueeze(0))
    with torch.no_grad():
      output = classifier(img)
      if len(train_frames) == 0:
        train_frames = output
      else:
        train_frames = torch.cat((train_frames, output), dim=0)
    if (i % 500) == 0:
      print(i)
  torch.save(train_frames, train_frames_path)

if not os.path.exists(test_frames_path):
  print("generating test frames representations")
  test_frames = []
  for i in range(len(test["aps_frame"])):
    img = np.flip(test["aps_frame"][i])
    img = torch.tensor(img.copy(), dtype=torch.float32).to(device)
    img = img.unsqueeze(0)
    img = img.repeat(3, 1, 1)
    img = img_transforms(img.unsqueeze(0))
    with torch.no_grad():
      output = classifier(img)
      if len(test_frames) == 0:
        test_frames = output
      else:
        test_frames = torch.cat((test_frames, output), dim=0)
    if (i % 500) == 0:
      print(i)
  torch.save(test_frames, test_frames_path)

## Model definition

window_size = 10

n_heads = 8
d_model = 160 #20 * n_heads

img_repr_dim = 1024

class PolicyNet(nn.Module):
  def __init__(self):
    super().__init__()
      
    self.img_ffn = nn.Sequential(
        nn.Linear(img_repr_dim, d_model),
        nn.ReLU()
    )
    self.sensor_ffn = nn.Sequential(
        nn.Linear(3, d_model),
        nn.ReLU()
    )
    #self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
    self.linear_q = nn.Linear(d_model, d_model)
    self.linear_v = nn.Linear(d_model, d_model)
    self.linear_k = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(p=0.1)
    self.linear_o = nn.Linear(d_model, d_model)

    self.ffn = nn.Sequential(nn.Linear(d_model, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(1024, d_model)
                            )

    self.norm1 = nn.LayerNorm(d_model, eps=1e-5, bias=True)
    self.norm2 = nn.LayerNorm(d_model, eps=1e-5, bias=True)
    self.dropout1 = nn.Dropout(0.1)
    self.dropout2 = nn.Dropout(0.1)
    self.head = nn.Sequential(
        nn.Linear(d_model, 3),
        nn.Sigmoid()
    )


  def forward(self, imgs, sensors):
    imgs = self.img_ffn(imgs)
    sensors = self.sensor_ffn(sensors)
    scaling = float(d_model // n_heads) ** -0.5
    x = torch.cat((imgs, sensors), dim=0)
    
    # generate q, v, and k
    q = self.linear_q(x)
    v = self.linear_v(x)
    k = self.linear_k(x[-1])

    q = q * scaling

    q = q.contiguous().view(-1, n_heads, d_model//n_heads).transpose(0,1) #[nheads, n_input_tokens, d_model//n_heads]
    v = v.contiguous().view(-1, n_heads, d_model//n_heads).transpose(0,1)
    k = k.contiguous().view(1, n_heads, d_model//n_heads).transpose(0,1)

    attn_weights = torch.matmul(q, k.transpose(1, 2)) # output shape is [n_heads, n_input_tokens, 1]

    attn_weights = attn_weights.transpose(1, 2)
    
    attn_weights = attn_weights.softmax(-1)

    attn_weights = self.dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, v).transpose(0,1).view(d_model)

    attn_output = self.linear_o(attn_output)

    attn_output = self.dropout1(attn_output)

    x = torch.add(x[-1], attn_output)

    x = self.norm1(x)

    ffn_output = self.ffn(x)

    ffn_output = self.dropout2(ffn_output)

    x = torch.add(x, ffn_output)

    x = self.norm2(x)

    x = self.head(x)

    return x

policyNet=PolicyNet()
policyNet.to(device)
print("model defined...")

## Training

print("training...")
train_handler = h5py.File('./rec1501614399_export.hdf5', 'r')

train=pd.DataFrame(columns=['accelerator_pedal_position',
                                       'brake_pedal_status',
                                       'steering_wheel_angle'])

train['accelerator_pedal_position'] = list(train_handler['accelerator_pedal_position'])
train['steering_wheel_angle'] = list(train_handler['steering_wheel_angle'])
train['brake_pedal_status'] = list(train_handler['brake_pedal_status'])

criterion = nn.L1Loss()
optimizer = optim.SGD(policyNet.parameters(), lr=0.01)

alpha = 0.2

num_epochs = 300
losses = []
train_frames = torch.load(train_frames_path)

for epoch in range(num_epochs):
  #initializing queues
  img_queue = []
  sensors_queue = []
  for i in range(len(train["accelerator_pedal_position"])):

    sensors = [train['accelerator_pedal_position'][i]/100,
                  (train['steering_wheel_angle'][i]+600)/1200,
                  train['brake_pedal_status'][i]]

    img = train_frames[i].squeeze(0)


    if ((len(img_queue) < (window_size-1)) and (len(sensors_queue) < (window_size-1))):
      img_queue.append(img.tolist())
      sensors_queue.append(sensors)
      continue
    elif ((len(img_queue) == (window_size-1)) and (len(sensors_queue) == (window_size-1))):
      img_queue.append(img.tolist())
      sensors_queue.append(sensors)
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
    loss = loss - alpha * torch.abs(sensors_queue_tensor[-1][2] - sensors_queue_tensor[-2][2])
    loss = torch.nn.functional.relu(loss)

    if(output.shape != sensors_queue_tensor[-1].shape):
      print(output.shape, sensors_queue_tensor[-1].shape)
      break
        
    loss.backward()
    
    optimizer.step()
    if(epoch == num_epochs-1):
      losses.append(loss.item())
    
  print(f"done epoch {epoch}.")

plt.figure()
plt.plot(losses)
plt.xlabel("samples")
plt.ylabel("losses")
plt.savefig("last_epoch_loss.png")

torch.save(policyNet.state_dict(), "./swin_weights_pi.pt")

## Testing

print("testing...")
test_handler = h5py.File('./rec1501612590_export.hdf5', 'r')

test=pd.DataFrame(columns=['accelerator_pedal_position',
                                       'brake_pedal_status',
                                       'steering_wheel_angle'])

test['accelerator_pedal_position'] = list(test_handler['accelerator_pedal_position'])
test['steering_wheel_angle'] = list(test_handler['steering_wheel_angle'])
test['brake_pedal_status'] = list(test_handler['brake_pedal_status'])

policyNet=PolicyNet()

policyNet.load_state_dict(torch.load("./swin_weights_pi.pt", weights_only = True))
policyNet.to(device)
policyNet.eval()

criterion = nn.L1Loss()

losses = []
predictions = []
test_frames = torch.load(test_frames_path)
with torch.no_grad():
  #initializing queues
  img_queue = []
  sensors_queue = []
  for i in range(len(test["accelerator_pedal_position"])):

    sensors = [test['accelerator_pedal_position'][i]/100,
                  (test['steering_wheel_angle'][i]+600)/1200,
                  test['brake_pedal_status'][i]]


    img = test_frames[i].squeeze(0)


    if ((len(img_queue) < (window_size-1)) and (len(sensors_queue) < (window_size-1))):
      img_queue.append(img.tolist())
      sensors_queue.append(sensors)
      continue
    elif ((len(img_queue) == (window_size-1)) and (len(sensors_queue) == (window_size-1))):
      img_queue.append(img.tolist())
      sensors_queue.append(sensors)
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

    loss = criterion(output, sensors_queue_tensor[-1])
    loss = loss - alpha * torch.abs(sensors_queue_tensor[-1][2] - sensors_queue_tensor[-2][2])
    loss = torch.nn.functional.relu(loss)
        
    predictions.append(output.tolist())

    losses.append(loss.item())
    if (i % 1000) == 0:
      print(i)

plt.figure()
plt.plot(losses)
plt.xlabel("samples")
plt.ylabel("loss")
plt.savefig("test_loss.png")

predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
torch.save(predictions_tensor, "./swin_predictions_pi.pt")
