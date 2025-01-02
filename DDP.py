import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import torch.distributed as dist
import os
#print('done')

rank = 0  
world_size = 2
master_addr = '127.0.0.1'
master_port = '29500'

os.environ['RANK'] = str(rank)
os.environ['WORLD_SIZE'] = str(world_size)
os.environ['MASTER_ADDR'] = master_addr
os.environ["MASTER_PORT"] = master_port


local_rank = rank % torch.cuda.device_count()


dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


batch_sizes = [1024 // world_size]

print('The batch size is {0}'.format(batch_sizes))


device = torch.device('cuda', local_rank)

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

print(f'Node {rank} - Running for different batch sizes:')
for batch_size in batch_sizes:
    print(f'Node {rank} - Batch Size: {batch_size}')

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

  
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2)

    resnet = models.resnet34(num_classes=10).to(device)

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
        print(f'Node {rank} - Batch Size: {batch_size}, Epoch {epoch + 1}, Loss: {running_loss / (i + 1):.4f}, Throughput: {throughput:.2f} images/second')

    print(f'Node {rank} - Batch Size: {batch_size} - Finished Training')


