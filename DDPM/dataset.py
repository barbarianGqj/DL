import torchvision
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader

def download_dataset():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download = True)
    # test the dataset
    # print("length of the dataset", len(mnist))
    # tmp = 4
    # img, label = mnist[tmp]
    # print("img", img)
    # print("label", label)
    # img.show()

def get_dataloader(batch_size):
    transform = Compose([ToTensor(), Lambda(lambda x: (x-0.5) * 2)])  # ToTensor()将0-255自动转成0-1的张量
    dataset = torchvision.datasets.MNIST(root='./data/mnist', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_img_shape():
    return (1, 28, 28)

if __name__ == '__main__':
    import os
    os.makedirs('work_dirs', exist_ok=True)
    download_dataset()
