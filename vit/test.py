import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights
import math

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_handler = h5py.File('../rec1501612590_export.hdf5', 'r')

test=pd.DataFrame(columns=['accelerator_pedal_position',
                                       'brake_pedal_status',
                                       'steering_wheel_angle'])

test['accelerator_pedal_position'] = list(test_handler['accelerator_pedal_position'])
test['steering_wheel_angle'] = list(test_handler['steering_wheel_angle'])
test['brake_pedal_status'] = list(test_handler['brake_pedal_status'])

window_size = 10
img_repr_dim = 768

n_heads = 8
d_model = 160 #20 * n_heads

alpha = 0.2

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

print("model defined...")

policyNet.load_state_dict(torch.load("./weights300.pt", weights_only = True))
policyNet.to(device)
policyNet.eval()

criterion = nn.L1Loss()

losses = []
predictions = []
test_frames = torch.load("./test_frames.pt")
with torch.no_grad():
  #initializing queues
  img_queue = []
  sensors_queue = []
  for i in range(len(test["accelerator_pedal_position"])):

    sensors = [test['accelerator_pedal_position'][i]/100,
                  test['brake_pedal_status'][i],
                  (test['steering_wheel_angle'][i]+600)/1200]


    img = test_frames[i].squeeze(0)


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

    loss = criterion(output, sensors_queue_tensor[-1])

    loss = loss - alpha * torch.abs(sensors_queue_tensor[-1][1] - sensors_queue_tensor[-2][1])
    loss = torch.nn.functional.relu(loss)

    predictions.append(output.tolist())

    losses.append(loss.item())
    if (i % 1000) == 0:
      print(i)

# saving losses

print(f"avg loss: {np.mean(losses)}")
print(f"max loss: {max(losses)}")
print(f"min loss: {min(losses)}")

plt.plot(losses)
plt.axhline(np.mean(losses), color='r')
plt.xlabel("samples")
plt.ylabel("loss")
plt.savefig("./test_loss.png")
print("losses plot saved as test_loss.png")

# saving accelerations
plt.figure()
accelerations = [element[0] for element in predictions]
truth = test['accelerator_pedal_position'][window_size-1:].tolist()
truth = [x / 100 for x in truth]
plt.plot(truth, color='r', label='true')
plt.plot(accelerations, color='b', label='predicted')
plt.xlabel('samples')
plt.ylabel('accelerator')
plt.legend()
plt.savefig("acceleration.png")
print("accelerations plot saved as acceleration.png")

# saving brakings
plt.figure()
brakes = [element[1] for element in predictions]
truth = test['brake_pedal_status'][window_size-1:].tolist()
plt.plot(truth, color='r', label='true')
plt.plot(brakes, color='b', label='predicted')
plt.xlabel('samples')
plt.ylabel('brake pedal')
plt.legend()
plt.savefig("brakes.png")
print("brakes plot saved as brakes.png")

# saving steerings
plt.figure()
steers = [element[2] for element in predictions]
truth = test['steering_wheel_angle'][window_size-1:].tolist()
truth = [(x + 600) / 1200 for x in truth]
plt.plot(truth[:], color='r', label='true')
plt.plot(steers[:], color='b', label='predicted')
plt.xlabel('samples')
plt.ylabel('steering wheel')
plt.legend()
plt.savefig("steerings.png")
print("steerings plot saves as steerings.png")

predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
torch.save(predictions_tensor, "./pn_predictions_300.pt")
