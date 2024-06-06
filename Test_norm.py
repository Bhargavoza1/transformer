import torch
import torch.nn as nn


def custom_std(x, dim , keepdim=True):
    # Compute the mean along the specified dimension
    mean = torch.mean(x, dim=dim, keepdim=True)

    # Compute the squared differences from the mean
    squared_diff = torch.square(x - mean)


    mean_squared_diff = torch.mean(squared_diff, dim=dim, keepdim=True)

    # Compute the square root to get the standard deviation
    std = torch.sqrt(mean_squared_diff)

    return std

# Define the batch normalization layer
batch_norm = nn.BatchNorm1d(3)

# Example input tensor (batch_size, num_features)
input_tensor = torch.tensor([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9],], dtype=torch.float32)
# Apply batch normalization
normalized_tensor = batch_norm(input_tensor)

print("Input Tensor:\n", input_tensor)
print("batch_norm Normalized Tensor:\n", normalized_tensor)



class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Initialize non-learnable parameters
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # Check if the model is in training mode
        if self.training:
            # Compute batch statistics
            batch_mean = torch.mean(x, dim=0,  keepdim = True)
            batch_std = custom_std(x, dim=0)
            batch_std2  = torch.std(x , dim=0 , keepdim = True)

            # Update running statistics using momentum
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_std
        else:
            # Use running statistics during inference
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Normalize the input
        x_normalized = (x - batch_mean) / (batch_std + self.eps)

        # Scale and shift
        scaled_shifted = self.weight * x_normalized + self.bias

        return scaled_shifted

my_bn = BatchNorm1d(3)

normalized_tensor = my_bn(input_tensor)
print("custom  BatchNorm1d Tensor:\n", normalized_tensor)


layer_norm = nn.LayerNorm(3)


# Apply layer normalization
normalized_tensor = layer_norm(input_tensor)

print("Input Tensor:\n", input_tensor)
print("layer_norm Normalized Tensor:\n", normalized_tensor)



class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = custom_std(x ,dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        normalized_tensor = (x - mean) / (std + self.eps)

        # Step 4: Scale the normalized tensor by alpha and shift it by bias
        scaled_shifted_tensor = self.alpha * normalized_tensor + self.bias

        # Step 5: Return the scaled and shifted tensor
        return scaled_shifted_tensor


my_lr = LayerNormalization(3)

normalized_tensor = my_lr(input_tensor)
print("custom  LayerNormalization Tensor:\n", normalized_tensor)

