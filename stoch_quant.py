import torch
import numpy as np



def sq_compress(vector,max_val,min_val,bits):
    #calculate the interval lengths
    interval = (max_val-min_val)/(2**bits-1)
    #calculate the index of the sample point to the left of each entry
    indices = torch.floor((vector-min_val)/interval)
    #calculate the probability of each entry being assigned to the left sample point based on the distance from it
    probs = (interval-(vector-min_val)+indices*interval)/interval
    #generate vector of random values between 0 and 1
    random_probs = torch.rand(vector.shape,device=vector.device)
    #add ones to the index assigned if the random value at the entry is greater than the assignment probability
    compressed_vec = indices + torch.ge(random_probs,probs).int()
    return compressed_vec

def sq_decompress(compressed_vector,max_val,min_val,bits,clients):
    #retrieve the value of the endpoint at the given index for each compressed entry
    decompressed_vector = ((compressed_vector*(max_val-min_val))/(clients*(2**bits-1)))+min_val
    return decompressed_vector

#######################################################################################################
############################## Information extraction #################################################
#######################################################################################################

def get_sq_compressable_extraction(vector,p):
    #extract the indices of len(vector)*p number of entries with highest absolute value
    topk,indices = torch.topk(torch.abs(vector),k=int(len(vector)*p))
    extreme = torch.zeros(len(vector),device=vector.get_device())
    #store extreme values its own vector at corresponding indices
    extreme[indices] = vector[indices]
    #0 out index in original vector containing extreme values
    filtered_vec = vector-extreme
    return filtered_vec,extreme


