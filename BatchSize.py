import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time

## Wrote the code based on resnet34 pytorch tutorial https://www.pluralsight.com/guides/introduction-to-resnet

batch_sizes = [64, 128, 256, 512, 1024]
print('here')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
print('Running for different batch sizes:')
for batch_size in batch_sizes:
    print(f'Batch Size: {batch_size}')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    resnet = models.resnet34(num_classes=10) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet.to(device)
    resnet.train()

    num_epochs = 5
    start_time = time.time()
    total_images = 0 

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.cuda.synchronize()
            optimizer.step()

            running_loss += loss.item()
            total_images += inputs.size(0)

        end_time = time.time()
        elapsed_time = end_time - start_time
        throughput = total_images / elapsed_time
        print(f'Batch Size: {batch_size}, Epoch {epoch + 1}, Loss: {running_loss / (i + 1):.4f}, Throughput: {throughput:.2f} images/second')

    print(f'Batch Size: {batch_size} - Finished Training\n')
