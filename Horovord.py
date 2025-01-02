import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import horovod.torch as hvd

batch_sizes = [1024]


hvd.init()
torch.cuda.set_device(hvd.local_rank())

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
print('Running for different batch sizes:')
start = time.time()
for batch_size in batch_sizes:
    print(f'Batch Size: {batch_size}')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

   
    batch_size = batch_size
    print('Horovord size is {0}'.format(hvd.size()))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=hvd.size(), rank=hvd.rank()
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler
    )

    resnet = models.resnet34(num_classes=10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    num_epochs = 5
    start_time = time.time()
    total_images = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        train_sampler.set_epoch(epoch)
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
end  = time.time()
print('Total time taken is {0}'.format(end-start))