# This file is for some functions that will be used for the implimentation of the algorithm. 
#Typically you will find the implimentation of projection into simplex function.


import numpy as np
import torch
from torch import nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def project_onto_simplex(input_coeffs):
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



def objctif_function(A,b):
        '''
        This function calcultes the euclidian distance between Ax and b 
        '''
        return lambda x: torch.norm(A.mm(x)-b).item()
    
def grad_obj_function(A,b):
        '''
        This function calculates the gradient of the euclidian distance. We use the close form of the gradient.
        '''
        return lambda x: (A.t().mm(A)).mm(x) - (A.t().mm(b))
    
def spectral_radius_AA_T(A):
        '''
        This function calculates the spectral radius of the matrix A*A^T
        '''
        m,n=A.size()
        if m<2000 and n<2000:
            # If the matrix is not too big, we claculate the true value of the spectral radius which corresponds to the maximum eigenvalue of A*A^T.
            return torch.max(torch.abs(torch.eig(A.mm(A.t()))[0][:,0]))
        else:
            # If the matrix is too big, we calculate an approximation of the spectral radius.
                y = torch.normal(0, torch.ones(n,1)).to(device)
                return( torch.norm(A.t().mm(A.mm(y)))/torch.norm(y))


def get_poison_tuples(poison_batch, poison_label):
    """
    Includes the labels
    """
    poison_tuple = [(poison_batch.poison.data[num_p].detach().cpu(), poison_label) for num_p in range(poison_batch.poison.size(0))]

    return poison_tuple

def get_poison_tuples(poison_batch, poison_label):
    """
    Includes the labels
    """
    poison_tuple = [(poison_batch.poison.data[num_p].detach().cpu(), poison_label) for num_p in range(poison_batch.poison.size(0))]

    return poison_tuple


def fetch_target(target_label, target_index, start_idx, path, subset, transforms):
    """
    Fetch the "target_index"-th target, counting starts from start_idx
    """
    img_label_list = torch.load(path)[subset]
    counter = 0
    for idx, (img, label) in enumerate(img_label_list):
        if label == target_label:
            counter += 1
            if counter == (target_index + start_idx + 1):
                if transforms is not None:
                    return transforms(img)[None, :,:,:]
                else:
                    return np.array(img)[None,:,:,:]
    raise Exception("Target with index {} exceeds number of total samples (should be less than {})".format(
                            target_index, len(img_label_list)/10-start_idx))


def load_pretrained_net(net_name, chk_name, model_chk_path, test_dp=0):
    """
    Load the pre-trained models. CUDA only :)
    """
    net = eval(net_name)(test_dp=test_dp)
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    print('==> Resuming from checkpoint for %s..' % net_name)
    checkpoint = torch.load('./{}/{}'.format(model_chk_path, chk_name) % net_name)
    if 'module' not in list(checkpoint['net'].keys())[0]:
        # to be compatible with DataParallel
        net.module.load_state_dict(checkpoint['net'])
    else:
        net.load_state_dict(checkpoint['net'])

    return net


class Poisonlist(torch.nn.Module):
    """
    This Class, inhirates from torch.nn.Modul. It creates a batch of learnable parameters from a list of tensors, and provides a method to access these parameters to simplify the optimization process.
    """
    def __init__(self, input_tensors):
        super(Poisonlist, self).__init__()
        base_list = torch.stack(input_tensors, 0)
        self.venom = torch.nn.Parameter(base_list.clone())

    def forward(self):
        return self.venom 
