import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
import numpy as np
from matplotlib import pyplot as plt
import h5py

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = h5py.File('../rec1501614399_export.hdf5', 'r')
test = h5py.File('../rec1501612590_export.hdf5', 'r')

window_size = 10

transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((200, 88)),
    ])

class CNN_LSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=5, stride=2),
    nn.Dropout2d(0.2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3),
    nn.Dropout2d(0.2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=2),
    nn.Dropout2d(0.2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3),
    nn.Dropout2d(0.2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=2),
    nn.Dropout2d(0.2),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128,128, kernel_size=3),
    nn.Dropout2d(0.2),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, 256, kernel_size=3),
    nn.Dropout2d(0.2),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=3),
    nn.Dropout2d(0.2),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(8192, 512),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.Dropout(0.5),
    nn.ReLU(),
    )

    self.lstm = nn.LSTM(input_size=512, hidden_size=32, num_layers=3)
    self.output = nn.Sequential(
        nn.Linear(32*window_size, 3)
    )

  def forward(self, x):
    x = self.cnn(x)
    x, h = self.lstm(x)
    x = x.reshape(-1)
    x = self.output(x)
    return x

cnnlstm = CNN_LSTM()
cnnlstm.to(device)

criterion = nn.L1Loss()
optimizer = optim.SGD(cnnlstm.parameters(), lr=0.01)

num_epochs = 300
queue_tens = None
losses = []
for epoch in range(num_epochs):
  for i in range(len(train["aps_frame"])):
    img = np.flip(train["aps_frame"][i])
    img = transform(img.copy())
    img.to(device)
    img = img.unsqueeze(0)
    if queue_tens == None:
      queue_tens = img
      continue
    elif len(queue_tens) < window_size:
      queue_tens = torch.cat((queue_tens, img), dim=0)
      continue

    queue_tens = torch.cat((queue_tens[1:], img), dim=0)
    if len(queue_tens) > window_size:
      print("window size exceeded")
      break
    label = torch.tensor([train['accelerator_pedal_position'][i]/100,
                train['brake_pedal_status'][i],
                (train['steering_wheel_angle'][i]+600)/1200], dtype=torch.float32).to(device)

    output = cnnlstm(queue_tens.to(device))
    optimizer.zero_grad()
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
  print(f"done for epoch {epoch}")
  if ((epoch +1)%50) == 0:
    torch.save(cnnlstm.state_dict(), f"./weights{epoch + 1}.pt")

print("training ended, starting test")

cnnlstm.eval()
queue_tens = None
losses = []
predictions = []
with torch.no_grad():
  for i in range(len(test["aps_frame"])):
    img = np.flip(test["aps_frame"][i])
    img = transform(img.copy())
    img.to(device)
    img = img.unsqueeze(0)
    if queue_tens == None:
      queue_tens = img
      continue
    elif len(queue_tens) < window_size:
      queue_tens = torch.cat((queue_tens, img), dim=0)
      continue

    queue_tens = torch.cat((queue_tens[1:], img), dim=0)
    if len(queue_tens) < window_size:
      print("window size exceeded")
      break
    label = torch.tensor([train['accelerator_pedal_position'][i]/100,
                train['brake_pedal_status'][i],
                (train['steering_wheel_angle'][i]+600)/1200], dtype=torch.float32).to(device)

    output = cnnlstm(queue_tens.to(device))
    loss = criterion(output, label)
    losses.append(loss.item())
    predictions.append(output.tolist())

print(f"avg loss: {np.mean(losses)}")
print(f"max loss: {max(losses)}")
print(f"min loss: {min(losses)}")

accelerations = [element[0] for element in predictions]
truth = test['accelerator_pedal_position'][window_size-1:].tolist()
truth = [x / 100 for x in truth]
accelerations = [x for x in accelerations]
plt.plot(truth, color='r', label='true')
plt.plot(accelerations, color='b', label='predicted')
plt.xlabel('samples')
plt.ylabel('accelerator')
plt.legend()
plt.savefig("./acc_cnnlstm_300.png")
plt.figure()

brakes = [element[1] for element in predictions]
truth = test['brake_pedal_status'][window_size-1:].tolist()
plt.plot(truth, color='r', label='true')
plt.plot(brakes, color='b', label='predicted')
plt.xlabel('samples')
plt.ylabel('brake pedal')
plt.legend()
plt.savefig("./brk_cnnlstm_300.png")
plt.figure()

steers = [element[2] for element in predictions]
truth = test['steering_wheel_angle'][window_size-1:].tolist()
truth = [(x + 600) / 1200 for x in truth]
steers = [x for x in steers]
plt.plot(truth[:], color='r', label='true')
plt.plot(steers[:], color='b', label='predicted')
plt.xlabel('samples')
plt.ylabel('steering wheel')
plt.legend()
plt.savefig("./str_cnnlstm_300.png")
plt.figure()

predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
torch.save(predictions_tensor, "./predictions_cnnlstm_300.pt")
