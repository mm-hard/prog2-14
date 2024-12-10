import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

#print(len(ds_train))

for i in range(12):
    image,target=ds_train[i]
    #print(type(image),target)
    plt.subplot(3,4,i+1)
    plt.imshow(image)
    plt.title(target)
#plt.show()

image_tensor=transforms.functional.to_image(image)
image_tensor=transforms.functional.to_dtype(image_tensor,dtype=torch.float32,scale=True)
print(image_tensor.shape,image_tensor.dtype)
print(image_tensor.min(),image_tensor.max())