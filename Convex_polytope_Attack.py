
import torch
from torch import nn
import warnings
import help_functions
from help_functions import project_onto_simplex,objctif_function,grad_obj_function,spectral_radius_AA_T,get_nearest_poison
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Convex_polytop_attack(torch.nn.Module):

    def __init__(self,pre_trained_model,target_img,poison_label,subset,
                 optimization_method,std,mean,poison_nearest_neighbor_tensor=None,path_to_database='',initialization_poison=None , poison_max_iter=5000,
                 decay_start=1e5,decay_end=2e6,learning_rate_optim=0.01,momentum=0.9 ,tol=1e-6,verbose=False, 
                 device=device,nbr_of_neighbours_for_poison=5,number_of_tensors_per_class_for_poison_feteching=10,transforms=transforms):
        '''
        The pre_trainet_model is the model that we want to attack. It can also correspond to a set of models 
        to which we want to transfer the attack. In the paper it corresponds to the set of phi^(i) that we will loop over
        in order to creat A at each step of the inner loop.
        The target_img is the set of clean images that we will modify to generate the poison images.
        initialization_poison is a an array of the stat of the poisning it is usefull from a practical point of view.
        mean and std are the mean and the standard deviation of the dataset on which the model was trained.
        '''
        super(Convex_polytop_attack, self).__init__()

        self.pre_trained_model = pre_trained_model
        self.target_img = target_img
        self.initialization_poison=initialization_poison
        self.tol = tol
        self.verbose = verbose
        self.device = device
        self.optimization_method=optimization_method
        self.learning_rate_optim=learning_rate_optim
        self.momentum=momentum
        self.poison_nearest_neighbor_tensor=poison_nearest_neighbor_tensor
        self.poison_max_iter=poison_max_iter
        self.dacay_start=decay_start
        self.decay_end=decay_end
        self.std=(torch.Tensor(std).reshape(1, 3, 1, 1)).to(self.device)
        self.mean=torch.Tensor(mean).reshape(1, 3, 1, 1).to(self.device)
        self.path_to_database=path_to_database
        self.nbr_of_neighbours_for_poison=nbr_of_neighbours_for_poison
        self.poison_label=poison_label
        self.number_of_tensors_per_class_for_poison_feteching=number_of_tensors_per_class_for_poison_feteching
        self.subset=subset
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std),])

        if path_to_database=='' and poison_nearest_neighbor_tensor==None:
            raise ValueError("You must provide a path to the database or a tensor of nearest neighbors of the target images.")
        elif path_to_database!='':
            self.poison_nearest_neighbor_tensor,indexes=get_nearest_poison(models_list=self.pre_trained_model, 
                    target=self.target_img, num_poison=self.nbr_of_neighbours_for_poison, 
                    poison_label=self.poison_label, num_per_class=self.number_of_tensors_per_class_for_poison_feteching, 
                    subset=self.subset,transforms=self.transform,data_base_path=self.path_to_database,device=self.device)
        

        self.poison_nearest_neighbor_tensor=[tensor.to('cuda') for tensor in self.poison_nearest_neighbor_tensor]                      
        self.initialization_poison=self.poison_nearest_neighbor_tensor

        self.nbr_of_poisons = len(self.poison_nearest_neighbor_tensor)
        self.poison_nearest_neighbor_tensor_concat = torch.stack(self.poison_nearest_neighbor_tensor, 0)
        self.poison_list = help_functions.Poisonlist(self.initialization_poison).to(self.device) # We create a batch of learnable parameters from a list of tensors, and provides a method to access these parameters to simplify the optimization process.
        self.base_range_m_batch = self.poison_nearest_neighbor_tensor_concat * std + mean
        self.target_img=self.target_img.to(self.device)







        if not isinstance(self.optimization_method, str):
            raise ValueError("The optimization method should be a string")
        elif not(self.optimization_method.lower() in ['sgd','signedadam','adam']):
            raise ValueError("The optimization method should be SGD, SignedAdam or Adam")
        else:
            print(f'The optimization method is {self.optimization_method}')
            if self.optimization_method.lower()=='sgd':
                self.optimizer=torch.optim.SGD(self.poison_list.parameters(), lr=self.learning_rate_optim, momentum=self.momentum)
            elif self.optimization_method.lower()=='signedadam':
                self.optimizer=torch.optim.SignedAdam(self.poison_list.parameters(), lr=self.learning_rate_optim)
            elif self.optimization_method.lower()=='adam': 
                self.optimizer=torch.optim.Adam(self.poison_list.parameters(), lr=self.learning_rate_optim)
        
        if self.poison_max_iter > self.dacay_start and self.poison_max_iter < self.decay_end:
            raise warnings.warn(f'''The number of iterations is big.\n The learning rate will be decayed 
                                starting from the {self.dacay_start} th ittration. 
                                \n You can change the decay_start parameter to a bigger value if you want.''')
        
        if self.poison_max_iter > self.decay_end:
            raise warnings.warn(f'''The number of iterations is big.
                                \n The learning rate will be decayed from
                                 the {self.dacay_start} th iteration  until 
                                 the {self.decay_end} th ittration. \n You can change the
                                decay_start and decay_end parameters to bigger values if you want.''')

        print(f"You are now ready to perform the attack. \n Here is a summary of the parameters that you have chosen: \n { vars(self)}")


    
    def step_inner_loop(A,b,x,i=0):
        '''
        This function calculates the step of the inner loop of the algorithm. 
        The A of this function is the same as the one defined in the paper. 
        However for the step we consider the spectral radius of A*A^T instead of the l_2 norm of A*A^T.
        The objective funciton calculated here is the euclidian distance between Ax and b. 
        Grad_f corresponds to the term A.T(Ac^(i)-phi^(i)(x_t)) in the paper so that we just compute a simple
        gradient descent step. Finally b corresponds to phi^(i)(x) in the paper and x corresponds to c^(i) in the paper.
        '''
        step=2/(spectral_radius_AA_T(A=A))
        f=objctif_function(A=A,b=b)
        grad_f=grad_obj_function(A=A,b=b)
        
        x_hat = x - step*grad_f(x)   # gradient descent  step
        if f(x_hat) > f(x):       
            step = step/2           # if the objective function strats increasing, we take a smaller step
            i==+1
            if i>20:
                warnings.warn("The Gradient Descent is not converging.")
        else:
            x_new = project_onto_simplex(x_hat)  
            x = x_new
            i=0
        return(x,i)


    def inner_loop(A,b,x_0,tol_inner,itter_max):
        '''
        We perform the inner loop of the algorithm.To avoid the problem of the gradient 
        descent not converging, we use a backtracking line search.
        We are just performing a loop with some stopping condition.
        '''
        x = x_0
        iter=0
        stopping_condition = False
        i=0
        while stopping_condition==False and iter<itter_max:
            iter+=1
            x_new,i_new=Convex_polytop_attack.step_inner_loop(A=A,b=b,x=x,i=i)
            i=i_new
            if i > 100:
                warnings.warn("The Gradient Descent is diverging the algorithm will quit the loop .")
                break
            if torch.norm(x-x_new)/max(torch.norm(x), 1e-8)<tol_inner:
                stopping_condition=True
            x=x_new
        return x
    

    def outer_loop(list_of_target_nets, list_of_target_featers, poison_batch, 
                   s_coeff_list,max_itter_inner ,tol_inner):
        """
        This function calculates the step of the outer loop of the algorithm.
        In this function list_of_target_nets corresponds to the set of phi^(i) that we will loop over
        in order to creat A at each step of the inner loop.
        """
        poison_network = [net(x=poison_batch(), penu=True) for net in list_of_target_nets]

        for i, (poison_features_vect, target_feat) in enumerate(zip(poison_network, list_of_target_featers)):
            s_coeff_list[i] = Convex_polytop_attack.inner_loop(A=poison_features_vect.t().detach(), b=target_feat.t().detach(),
                                                    x_0=s_coeff_list[i], tol=tol_inner,itter_max=max_itter_inner)

        total_loss = 0
        # Calculate the loss after one ster of the outer loop
        for net, s_coeff, target_feat, poison_feat_mat in zip(list_of_target_nets, s_coeff_list, 
                                                              list_of_target_featers, poison_network):

            residual = target_feat - torch.sum(s_coeff * poison_feat_mat, 0, keepdim=True)
            target_norm_square = torch.sum(target_feat ** 2)
            recon_loss = 0.5 * torch.sum(residual**2) / target_norm_square
            total_loss += recon_loss


        return total_loss, s_coeff_list
        
    




    def poisan_generation(self,decay_ratio=0.1,tol_inner=1e-6,max_itter_inner=10000,epsilon=0.1,pos_label_poison=-1):
        '''
        This function corresponds to the algorithm 1 of the paper.
        '''

        #self.target_pretrained_net.eval() # We put the target model in evaluation mode so that our attck is not trained on the target model.
        std, mean = std.to(self.device), mean.to(self.device)
        target_list = []
        s_init_coeff_list = []
        
        iter=0

        for l,net in enumerate(self.pre_trained_model): # We loop over the models that we want to attack.
            net.eval() # We put the model in evaluation mode so that our attck is not trained on the model.
            target_list.append(net(x=self.target_img, penu=True).detach())
            s_init_coeff_list.append(torch.ones(self.nbr_of_poisons, 1).to(self.device) / self.nbr_of_poisons)
        while iter<self.poison_max_iter and iter<self.decay_end+1:
            
            if iter==self.dacay_start:
                raise warnings.warn(f'''Starting from this iteration ie {iter} th iteration, the learning 
                                    rate will be decayed by {decay_ratio} at each itteration.\n You can change 
                                    by default parameters if you wish.''')

            if iter >  self.dacay_start and iter <  self.decay_end:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= decay_ratio
            if iter==self.decay_end:
                raise warnings.warn(f'''Starting from this iteration ie {iter} th iteration, the optimization procedure will stop.
                                    \n You can change the decay_end parameter if you wish.''')
            

            self.poison_list.zero_grad()
            total_loss, s_init_coeff_list = Convex_polytop_attack.outer_loop(A=self.pre_trained_model,b=target_list,poison_batch=self.poison_list, 
                                                       s_coeff_list=s_init_coeff_list,max_itter_inner=max_itter_inner ,tol_inner=tol_inner)
            total_loss.backward()
            self.optimizer.step()
            # cliping the poison images so that the infinity norm constraint is satisfied..
            perturb_range01 = torch.clamp((self.poison_list.venom.data - self.poison_nearest_neighbor_tensor_concat) * std, -epsilon, epsilon)
            perturbed_range01 = torch.clamp(self.base_range_m_batch.data + perturb_range01.data, 0, 1)
            self.poison_list.venom.data = (perturbed_range01 - self.mean) / self.std
            # Update the itteration
            iter+=1
        return help_functions.get_poison_tuples(self.poison_list, pos_label_poison), total_loss.item()
