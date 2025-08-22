import torch
import torch.nn as nn


class SimAM(torch.nn.Module):
    def __init__(self, channels = None,out_channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)  
    
    
    
# ------------------------------------------------------------------------
class SimAMEnhanced(nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4, activation_fn=None):
        super(SimAMEnhanced, self).__init__()
        
        self.e_lambda = e_lambda
        self.activation = activation_fn if activation_fn else nn.Sigmoid()
        self.epsilon = 1e-6  # Small value for numerical stability

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam_enhanced"

    def forward(self, x):
        b, c, h, w = x.size()

        # Space-wise attention (similar to original SimAM)
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        space_attention = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda + self.epsilon)) + 0.5

        # Channel-wise attention (Global average pooling)
        channel_attention = torch.mean(x, dim=[2, 3], keepdim=True)  # Global average pooling
        channel_attention = self.activation(channel_attention)

        # Combine space and channel attention
        attention_map = space_attention * channel_attention
        return x * self.activation(attention_map)
#----------------------------------------------------------------------
class SimAMAdaptive(nn.Module):
    def __init__(self, channels=None, out_channels=None, activation_fn=None):
        super(SimAMAdaptive, self).__init__()
        
        # Learnable lambda parameter
        self.e_lambda = nn.Parameter(torch.tensor(1e-4))  # lambda is now learnable
        self.activation = activation_fn if activation_fn else nn.Sigmoid()
        self.epsilon = 1e-6

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda.item())
        return s

    @staticmethod
    def get_module_name():
        return "simam_adaptive"

    def forward(self, x):
        b, c, h, w = x.size()

        # Space-wise attention
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        space_attention = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda + self.epsilon)) + 0.5

        # Adaptive attention based on specific location features (additional positional info can be added here)
        channel_attention = torch.mean(x, dim=[2, 3], keepdim=True)  # Global average pooling
        channel_attention = self.activation(channel_attention)

        # Combine spatial and channel attention
        attention_map = space_attention * channel_attention
        return x * self.activation(attention_map)

if __name__ == '__main__':
    from torchinfo import summary
    model = SimAMAdaptive()
    model = SimAMEnhanced()
    print("Check output shape ...")
    summary(model, input_size=(1, 3, 224, 224))
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    for i in y:
        print(i.shape)