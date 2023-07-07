import math
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as tdt
import torch.nn as nn
import torch.optim as optim
from ResNet import ResNet9,ResNet18,ResNet34,ResNet50,ResNet101,ResNet152

import os
import argparse
import pickle
import pkbar


from stoch_quant import sq_compress,sq_decompress,get_sq_compressable_extraction
from drive import drive_compress, drive_decompress
from fedavg import fedavg_compress, fedavg_decompress
import eden


##########################################################################
####################### Training #########################################
##########################################################################

device = torch.device("cuda")

##### from here to line 155 : https://github.com/amitport/DRIVE-One-bit-Distributed-Mean-Estimation/blob/main/experiments/distributed/distributed_cnn.py ####

def CIFAR10(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 10

    return num_classes, trainset, trainloader, testset, testloader

def CIFAR100(batch_size):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 100

    return num_classes, trainset, trainloader, testset, testloader

def MNIST(all_clients_batch_size):
    
    transform_train = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    transform_test = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train)   
    trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=all_clients_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    num_classes = 10

    return num_classes, trainset, trainloader, testset, testloader

def train(epoch):
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    
    compressed_client_grad_vecs = {}

    for batch_idx, (client_inputs, client_targets) in enumerate(trainloader):
        client_index = batch_idx % clients
        client_inputs, client_targets = client_inputs.to(device), client_targets.to(device)

        optimizer.zero_grad()
        client_outputs = net(client_inputs)
        client_loss = criterion(client_outputs, client_targets)
        client_loss.backward()

        ##############################################################
        ################## extract client gradient ###################
        ##############################################################

        client_grad_vec = []
        for param in net.parameters():
            x = param.grad.view(-1)
            client_grad_vec.append(x)

        client_grad_vec = torch.cat(client_grad_vec)
        ##############################################################
        ############## update client stats ###########################
        ##############################################################

        train_loss += client_loss.item()
        _, predicted = client_outputs.max(1)
        total += client_targets.size(0)
        correct += predicted.eq(client_targets).sum().item()

        ##############################################################
        ############## compress client gradients #####################
        ##############################################################
###### above is from: https://github.com/amitport/DRIVE-One-bit-Distributed-Mean-Estimation/blob/main/experiments/distributed/distributed_cnn.py ####
###### below is my own work #######
        if compression_alg == "drive":
            if need_padding:
                padded_client_grad_vec = torch.zeros(padded_dimension, device=device)
                padded_client_grad_vec[:params['dimension']] = client_grad_vec
                compressed_client_grad_vec = compress(padded_client_grad_vec,
                                                      prng=snd_prngs[client_index])
            else:
                compressed_client_grad_vec = compress(client_grad_vec,
                                                      prng=snd_prngs[client_index])
        elif compression_alg == "sq":
            #store client grad vec
            compressed_client_grad_vecs[client_index] = client_grad_vec
        elif compression_alg == "sq_1":
            compressed_client_grad_vecs[client_index] = get_sq_compressable_extraction(client_grad_vec,p)
        elif compression_alg == "fedavg":
            compressed_client_grad_vec = compress(client_grad_vec)
        elif compression_alg == "eden":
            if need_padding:
                padded_client_grad_vec = torch.zeros(padded_dimension, device=device)
                padded_client_grad_vec[:params['dimension']] = client_grad_vec
                compressed_client_grad_vec = compress(padded_client_grad_vec)
            else:
                compressed_client_grad_vec = compress(client_grad_vec)

        ##############################################################
        ############## append to clients gradient list ###############
        ##############################################################
        if compression_alg in ["fedavg","drive","eden"]:
            compressed_client_grad_vecs[client_index] = compressed_client_grad_vec
        ##############################################################
        ##############  finished a pass over all clients #############
        ##############################################################

        grad_vec = torch.zeros(params['dimension'], device=device)
        
        ##############################################################
        ############## all clients finished a batch? #################
        ##############################################################
        
        if client_index == clients - 1:
            ##################################################################
            ####### sq: compute max min and compress vecs ##############
            ##################################################################

            if compression_alg == "sq":
                max_val = -1000000000000
                min_val = 1000000000000
                for c in range(clients):
                    max_val = max(max_val,torch.max(compressed_client_grad_vecs[c]))
                    min_val = min(min_val,torch.min(compressed_client_grad_vecs[c]))
                for client in range(clients):
                    compressed_client_grad_vecs[client] = compress(compressed_client_grad_vecs[client],max_val,min_val,bits)
            elif compression_alg == "sq_1":
                max_val = -1000000000000
                min_val = 1000000000000
                for c in range(clients):
                    max_val = max(max_val,torch.max(compressed_client_grad_vecs[c][0]))
                    min_val = min(min_val,torch.min(compressed_client_grad_vecs[c][0]))
                for client in range(clients):
                    compressed_client_grad_vecs[client] = (compress(compressed_client_grad_vecs[client][0],max_val,min_val,bits),compressed_client_grad_vecs[client][1])
            ##################################################################
            ####### compute the total gradient: merge and decompress #########
            ##################################################################
            if compression_alg == "drive":
                for index, vec in compressed_client_grad_vecs.items():
                    ccgv, ccgv_metadata = vec
                    grad_vec += decompress(ccgv, ccgv_metadata,
                                            prng=rcv_prngs[index])[:params['dimension']]
            elif compression_alg == "sq":
                compressed_client_vecs_sum = torch.zeros(params['dimension'], device=device)
                for index,vec in compressed_client_grad_vecs.items():
                    compressed_client_vecs_sum += vec
                grad_vec = decompress(compressed_client_vecs_sum,max_val*clients,min_val*clients,bits,clients)
            elif compression_alg == "sq_1":
                compressed_client_vecs_sum = torch.zeros(params['dimension'], device=device)
                extreme_values = torch.zeros(params['dimension'], device=device)
                for index,client_data in compressed_client_grad_vecs.items():
                    compressed_client_vecs_sum += client_data[0]
                    extreme_values += client_data[1]
                grad_vec = decompress(compressed_client_vecs_sum,max_val*clients,min_val*clients,bits,clients) + extreme_values
            elif compression_alg == "fedavg":
                for index,vec in compressed_client_grad_vecs.items():
                    grad_vec += decompress(vec)
            elif compression_alg == "eden":
                for index,vec in compressed_client_grad_vecs.items():
                    cv,ctx = vec
                    grad_vec += decompress(cv,ctx)[0][:params['dimension']]

            #######################################################################################
            ####################################Compute grad vec###################################
            #######################################################################################

##### below is from : https://github.com/amitport/DRIVE-One-bit-Distributed-Mean-Estimation/blob/main/experiments/distributed/distributed_cnn.py ####           
            grad_vec /= clients
            

            ##################################################################
            ########## make sure dict is clean for next round ################
            ##################################################################

            compressed_client_grad_vecs = {}

            ##################################################################
            ################## write modified gradient #######################
            ##################################################################

            offset = 0
            for param in net.parameters():
                slice_size = len(param.grad.view(-1))
                grad_slice = grad_vec[offset:offset + slice_size]
                offset += slice_size

                y = torch.Tensor(grad_slice.to('cpu')).resize_(param.grad.shape)
                param.grad = y.to(device).clone()

            ##################################################################
            ################## step ##########################################
            ##################################################################

            optimizer.step()

    ### return acc
    epoch_train_acc = 100. * correct / total
    
    return epoch_train_acc


##########################################################################
####################### Testing ##########################################
##########################################################################

def test(epoch):
    
    global best_acc

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc or epoch == start_epoch + num_epochs:

        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_{0}.pth'.format(compression_alg))
        best_acc = acc

    ### return acc
    epoch_test_acc = 100. * correct / total

    return epoch_test_acc

#####above is from: https://github.com/amitport/DRIVE-One-bit-Distributed-Mean-Estimation/blob/main/experiments/distributed/distributed_cnn.py #### 

if __name__ == '__main__':

##### below is my own work #####
    
    ### clients and batch sizes
    clients=10
    clientBatchSize=128
    bits = 2
    p = 1/(2*6)
    compression_alg = "sq"
    model = "ResNet50"
    dataset = "CIFAR100"
    ##########################################################################
    ##########################################################################
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    ##########################################################################
    ##########################################################################

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    num_epochs = 30
    ##########################################################################
    ####################### Preparing data ###################################
    ##########################################################################
    if dataset == "CIFAR10":
        num_classes, trainset, trainloader, testset, testloader = CIFAR10(
            clientBatchSize)
    elif dataset == "CIFAR100":
        num_classes, trainset, trainloader, testset, testloader = CIFAR100(
            clientBatchSize)
    elif dataset == "MNIST":
        num_classes, trainset, trainloader, testset, testloader = MNIST(
            clientBatchSize)

    ##########################################################################
    ########################## Net ###########################################
    ##########################################################################

    print('==> Building model..')
    if model == "ResNet9":
        net = ResNet9(num_classes)
    elif model == "ResNet18":
        net = ResNet18(num_classes)
    elif model == "ResNet34":
        net = ResNet34(num_classes)
    elif model == "ResNet50":
        net = ResNet50(num_classes)
    elif model == "ResNet101":
        net = ResNet101(num_classes)
    elif model == "ResNet152":
        net = ResNet152(num_classes)

#### above is my own work ####
#### below is from: https://github.com/amitport/DRIVE-One-bit-Distributed-Mean-Estimation/blob/main/experiments/distributed/distributed_cnn.py ####

    ##########################################################################
    ########################## Cuda ##########################################
    ##########################################################################

    net = net.to(device)

    ##########################################################################
    ########################## Trainable parameters ##########################
    ##########################################################################

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    if (pytorch_total_params != pytorch_total_params_trainable):
        raise Exception("pytorch_total_params != pytorch_total_params_trainable")

    ##########################################################################
    ########################## Parameters ####################################
    ##########################################################################

    params = {}

    params['dimension'] = pytorch_total_params_trainable
    params['gradLayerLengths'] = []
    params['gradLayerDims'] = []

    for param in net.parameters():
        params['gradLayerDims'].append(param.size())
        params['gradLayerLengths'].append(len(param.view(-1)))

    padded_dimension = params['dimension']
    if not params['dimension'] & (params['dimension'] - 1) == 0:
        padded_dimension = int(2 ** (np.ceil(np.log2(params['dimension']))))
    need_padding = padded_dimension != params['dimension']

    ##########################################################################
    ########################## PRNGs #########################################
    ##########################################################################

    rcv_prngs = {}
    snd_prngs = {}

    for client_index in range(clients):
        seed = np.random.randint(2 ** 31)

        rgen = torch.Generator(device=device)
        rgen.manual_seed(seed)

        rcv_prngs[client_index] = rgen

        sgen = torch.Generator(device=device)
        sgen.manual_seed(seed)

        snd_prngs[client_index] = sgen

    ##########################################################################
    ############################### scheme ###################################
    ##########################################################################
    if compression_alg == 'drive':
        compress = drive_compress
        decompress = drive_decompress
    elif compression_alg == 'sq' or compression_alg == "sq_1":
        compress = sq_compress
        decompress = sq_decompress
    elif compression_alg == 'fedavg':
        compress = fedavg_compress
        decompress = fedavg_decompress
    elif compression_alg == "eden":
        edn = eden.eden_builder(bits=bits,device=device)
        compress = edn.forward
        decompress = edn.backward
    ##########################################################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    ##########################################################################
    ####################### Run ##############################################
    ########################################################################## 

    # collect stats to lists
    results = {
        'epochs': [],
        'trainACCs': [],
        'testACCs': [],
    }

    if not os.path.isdir('results'):
        os.mkdir('results')

    if not os.path.isdir('results/distributed_cnn'):
        os.mkdir('results/distributed_cnn')

    for epoch in range(start_epoch + 1, start_epoch + 1 + num_epochs):
        print("epoch: " + str(epoch))
        train_acc = train(epoch)
        test_acc = test(epoch)
        print("train acc: " + str(train_acc))
        results['epochs'].append(epoch)
        results['trainACCs'].append(train_acc)
        results['testACCs'].append(test_acc)
        if input() == "p":
            break

    with open('./results/distributed_cnn/' + 'results_' + compression_alg + '_' + dataset + '_' + model +'.pkl', 'wb') as filehandle:
        pickle.dump(results, filehandle)