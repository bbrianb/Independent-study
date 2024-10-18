from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn


class Model(nn.Module):
    def __init__(self, inputShape, hiddenUnits, outputShape):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inputShape, hiddenUnits),
            nn.Linear(hiddenUnits, outputShape)
        )
    
    def forward(self, x):
        return self.stack(x)

trainData = datasets.FashionMNIST('data', transform=ToTensor(), download=True)
testData = datasets.FashionMNIST('data', False, ToTensor(), download=True)
print(f'{len(trainData)=}, {len(testData)=}')
classNames = trainData.classes
print(f'{classNames=}')
image, label = trainData[0]
print(f'{label=}, {classNames[label]=}')
print(f'{image.shape=}')
plt.imshow(image.squeeze(), cmap='gray')
# plt.show()
batchSize = 32
trainDL = DataLoader(trainData, batchSize)
testDL = DataLoader(testData, batchSize)
print(f'{len(trainDL)=}, {len(testDL)=}')
firstBatchPictures, firstBatchLabels = next(iter(trainDL))
print(f'{firstBatchPictures.shape=}, {firstBatchLabels.shape=}')
picture = firstBatchPictures[0]
flatPicture = nn.Flatten()(picture)
print(f'{picture.shape=}, {flatPicture.shape=}, {28*28=}')
print(f'{len(classNames)=}')
model = Model(28*28, 10, len(classNames))