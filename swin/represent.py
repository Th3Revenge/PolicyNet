import torch
import torch.nn as nn
import h5py
import numpy as np
from torchvision.models import swin_b
from torchvision.models import Swin_B_Weights

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")
print(f"Version: {torch.__version__}, GPU: {torch.cuda.is_available()}, NUM_GPU: {torch.cuda.device_count()}")

train = h5py.File('../rec1501614399_export.hdf5', 'r')
test = h5py.File('../rec1501612590_export.hdf5', 'r')

classifier = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
classifier.head = nn.Identity()
classifier.to(device)
classifier.eval()
print("model downloaded and intialized...")
img_transforms = Swin_B_Weights.IMAGENET1K_V1.transforms()

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
torch.save(train_frames, './train_frames.pt')

print("generated train images representations in train_frames.pt...")

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
torch.save(test_frames, './test_frames.pt')

print("generated test images representations in test_frames.pt")


