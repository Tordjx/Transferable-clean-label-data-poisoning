
import torch
from torch import nn
import warnings





class Conex_polytop_attack(torch.nn.Module):
    '''
    '''
    def __init__(self,A, b, x_init, tol=1e-6, verbose=False, device='cuda'):
        '''
        '''
        super(Conex_polytop_attack, self).__init__()

        self.A = A
        self.b = b
        self.x_init = x_init
        self.tol = tol
        self.verbose = verbose
        self.device = device
        self.m,self.n=self.A.size()

        if self.x_int==None:
            self.x_init = torch.zeros(self.b.size(), device=self.device)
        elif torch.is_tensor(self.x_init)==False:
            self.x_init = torch.tensor(self.x_init, requires_grad=True)
        if torch.is_tensor(self.A)==False:
            self.A=torch.tensor(self.A, device=self.device)
        if torch.is_tensor(self.b)==False:
            self.b=torch.tensor(self.b, device=self.device)
        if b.size()[0] != A.size()[0]:
           raise RuntimeError(f"Dimensions of A and b do not match: {b.size()[0]} is different from {A.size()[0]}")
