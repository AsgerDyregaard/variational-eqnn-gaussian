# The random layer used in variational inference.
# I.e. the layer which replaces the nn.Linear in the nets.
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import nn
from torch.distributions.normal import Normal

class linear_var(nn.Module):

    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.mean = Parameter(torch.empty((out_features, in_features)))
        self.log_var = Parameter(torch.empty((out_features, in_features)))

        if bias:
            self.bias = Parameter(torch.normal(0,0.0005,size=(out_features,)))
            self.bias_log_var = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            torch.nn.init.normal_(self.mean,mean=0,std=0.001)
            torch.nn.init.normal_(self.log_var,mean = -9, std = 0.00001)
        
        if self.bias is not None:
            torch.nn.init.normal_(self.bias_log_var,mean = -9, std = 0.00001)
    
    def get_divergence(self):
        # Estimates the KL-divergence of this layer with a single sampling
        W_std = torch.sqrt(torch.exp(self.log_var))
        W = self.mean + W_std * torch.randn_like(self.log_var)

        kl_W = Normal(self.mean,W_std).log_prob(W).sum() - Normal(0,1).log_prob(W).sum()

        if self.bias is not None:
            b_std =  torch.sqrt(torch.exp(self.bias_log_var))
            b = self.bias + b_std * torch.randn_like(self.bias_log_var)
            kl_b = Normal(self.bias,b_std).log_prob(b).sum() - Normal(0,1).log_prob(b).sum()

            return kl_W + kl_b
        
        else:
            return kl_W
    
    def forward(self, x):
        # Local reparameterization trick from p. 114 in
        # "Variational Inference and Deep Learning A New Synthesis", Kingma, 2017.
        if self.bias is not None:
            gamma = torch.matmul(x,self.mean.mT) + self.bias
            delta = torch.matmul(torch.pow(x,2),torch.exp(self.log_var.mT)) + torch.exp(self.bias_log_var)

            return gamma + torch.sqrt(delta+ 1e-8) * torch.randn_like(gamma)
            
            # Sometimes an element of delta can become extremely small causing backpropagation to become impossible.
            # To combat this, the constant 1e-8 is added.
            # Sometimes the minimum value of delta is several order of magnitudes larger than this.
        
        else:
            gamma = torch.matmul(x,self.mean.mT)
            delta = torch.matmul(torch.pow(x,2),torch.exp(self.log_var.mT))

            return gamma + torch.sqrt(delta + 1e-8) * torch.randn_like(gamma)