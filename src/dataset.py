import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(config):
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    transform_train = transforms.Compose([
        transforms.RandomCrop((32,32), padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15), 
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
        transforms.RandomErasing(p=0.3, scale=(0.05, 0.2), value='random')
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader