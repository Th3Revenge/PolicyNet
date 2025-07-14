import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
import numpy as np
from matplotlib import pyplot as plt
import h5py
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = h5py.File('../rec1501614399_export.hdf5', 'r')
test = h5py.File('../rec1501612590_export.hdf5', 'r')

window_size = 10

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

img_transforms = ResNet50_Weights.IMAGENET1K_V1.transforms()

for parameter in model.parameters():
    parameter.requires_grad = False
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 3),
    nn.Sigmoid()
)

model.to(device)

criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []
num_epochs = 300
for epoch in range(num_epochs):
  for i in range(len(train["aps_frame"])):
    img = np.flip(train["aps_frame"][i])
    label = [train['accelerator_pedal_position'][i]/100,
                train['brake_pedal_status'][i],
                (train['steering_wheel_angle'][i]+600)/1200]
    label = torch.tensor(label, dtype=torch.float32).to(device)
    label = label.unsqueeze(0)
    img = torch.tensor(img.copy(), dtype=torch.float32).to(device)
    img = img.unsqueeze(0)
    img = img.repeat(3, 1, 1)
    img = img_transforms(img.unsqueeze(0))
    output = model(img)
    optimizer.zero_grad()
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
  print(f"done for epoch {epoch}")
  if ((epoch+1)%50)==0:
    torch.save(model.state_dict(), f"./weights{epoch+1}.pt")

print("training ended, starting test")

model.eval()
losses = []
predictions = []
with torch.no_grad():
  for i in range(len(test["aps_frame"])):
    img = np.flip(test["aps_frame"][i])
    label = [test['accelerator_pedal_position'][i]/100,
                test['brake_pedal_status'][i],
                (test['steering_wheel_angle'][i]+600)/1200]
    label = torch.tensor(label, dtype=torch.float32).to(device)
    label = label.unsqueeze(0)
    img = torch.tensor(img.copy(), dtype=torch.float32).to(device)
    img = img.unsqueeze(0)
    img = img.repeat(3, 1, 1)
    img = img_transforms(img.unsqueeze(0))
    output = model(img)
    loss = criterion(output, label)
    losses.append(loss.item())
    predictions.append(output.tolist())

print("test ended")

print(f"avg loss: {np.mean(losses)}")
print(f"max loss: {max(losses)}")
print(f"min loss: {min(losses)}")

accelerations = [element[0][0] for element in predictions]
truth = test['accelerator_pedal_position'][:].tolist()
truth = [x / 100 for x in truth]
accelerations = [x for x in accelerations]
plt.plot(truth, color='r', label='true')
plt.plot(accelerations, color='b', label='predicted')
plt.xlabel('samples')
plt.ylabel('accelerator')
plt.legend()
plt.savefig("./acc_resnet_300.png")
plt.figure()

brakes = [element[0][1] for element in predictions]
truth = test['brake_pedal_status'][:].tolist()
plt.plot(truth, color='r', label='true')
plt.plot(brakes, color='b', label='predicted')
plt.xlabel('samples')
plt.ylabel('brake pedal')
plt.legend()
plt.savefig("./brk_resnet_300.png")
plt.figure()

steers = [element[0][2] for element in predictions]
truth = test['steering_wheel_angle'][:].tolist()
truth = [(x + 600) / 1200 for x in truth]
steers = [x for x in steers]
plt.plot(truth[:], color='r', label='true')
plt.plot(steers[:], color='b', label='predicted')
plt.xlabel('samples')
plt.ylabel('steering wheel')
plt.legend()
plt.savefig("./str_resnet_300.png")

predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
torch.save(predictions_tensor, "./predictions_cnn_300.pt")
