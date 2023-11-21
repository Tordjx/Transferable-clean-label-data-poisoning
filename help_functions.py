# This file is for some functions that will be used for the implimentation of the algorithm. 
#Typically you will find the implimentation of projection into simplex function.


import numpy as np
import torch



def project_onto_simplex_torch(input_coeffs):
    """
    Project onto probability simplex.
    input_coeffs: Tensor
    """
    input_array = input_coeffs.view(-1).detach()
    sorted_array, indices = torch.sort(input_array, descending=True)
    cumulative_sum = torch.cumsum(sorted_array, dim=0) - 1
    rho = torch.arange(1, input_array.shape[0] + 1, device=input_coeffs.device)[(sorted_array - cumulative_sum / torch.arange(1, input_array.shape[0] + 1, device=input_coeffs.device)) > 0][-1]
    theta = cumulative_sum[rho - 1] / float(rho)
    projected_array = torch.nn.functional.relu(input_array - theta)
    return projected_array.view(input_coeffs.size())