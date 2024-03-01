import torch
import torch.nn.functional as F
from torch import Tensor, nn


def absmax_quantize(x: Tensor, bits: int = 8):
    """
    Absmax Quantization

    Args:
        x (torch.Tensor): Input tensor
        bits (int, optional): Number of bits. Defaults to 8.

    """
    Qb = 2 ** (bits - 1) - 1
    scale = Qb / torch.max(torch.abs(x))
    quant = (scale * x).round()
    dequant = quant / scale
    return quant.to(torch.int8), dequant


class BitLinear15b(nn.Module):
    """
    BitLinear implements a fully connected layer with ternary weight quantization.
    Weights are quantized to -1, 0, or +1 using an absmean quantization approach.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the BitLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.eps = 1e-6  # Small epsilon for numerical stability

    def forward(self, x):
        """
        Forward pass through the BitLinear layer.

        Args:
            x (Tensor): Input tensor of shape (..., in_features).

        Returns:
            Tensor: Output tensor of shape (..., out_features).
        """
        x = torch.sign(x)
        quantized_weight = self.quantize_weights(self.weight)
        return F.linear(x, quantized_weight)
        # return x

    def quantize_weights(self, W):
        """
        Quantizes the weights using the absmean quantization function.

        Args:
            W (Tensor): The weight tensor to be quantized.

        Returns:
            Tensor: Quantized weight tensor.
        """
        gamma = torch.mean(torch.abs(W)) + self.eps
        W_scaled = W / gamma
        W_quantized = torch.sign(W_scaled) * torch.clamp(
            torch.abs(W_scaled).round(), max=1.0
        )
        return W_quantized

    def extra_repr(self):
        """
        Provides additional information for debugging and logging.
        """
        return "in_features={}, out_features={}, quantization=ternary".format(
            self.in_features, self.out_features
        )


# # Initialize the BitLinear layer
# bit_linear = BitLinear15b(in_features=128, out_features=64)
# # Example input tensor
# x = torch.randn(10, 128)  # Example input
# output = bit_linear(x)  # Forward pass
# print(output)
    

class BitLinear(nn.Module):
    """
    BitLinear module as described in the BitNet architecture.

    This module performs a linear transformation with 1-bit quantized weights.
    The transformation includes a quantization step, matrix multiplication,
    and a subsequent dequantization step. Both the quantization and
    dequantization steps utilize learnable parameters gamma and beta.

    Attributes:
    - in_features: size of each input sample
    - out_features: size of each output sample
    - gamma: scaling factor for absmax quantization (learnable parameter)
    - beta: scaling factor for dequantization (learnable parameter)
    - weight: the 1-bit quantized weights of the linear transformation
    - bias: the bias term for the linear transformation (optional)
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the BitLinear module.

        Parameters:
        - in_features: An integer, the number of input features.
        - out_features: An integer, the number of output features.
        - bias: A boolean, whether the layer includes a bias.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

        # Learnable parameters for quantization and dequantization
        self.gamma = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, input):
        """
        Forward pass of the BitLinear module.

        Parameters:
        - input: A tensor of shape (batch_size, in_features).

        Returns:
        - output: A tensor of shape (batch_size, out_features).
        """
        # Apply Layer Normalization
        input_norm = F.layer_norm(input, (self.in_features,))

        # Absmax Quantization
        quant_scale = torch.max(torch.abs(input_norm), dim=1, keepdim=True).values
        input_quant = torch.sign(input_norm) * (quant_scale / self.gamma)

        # 1-bit Weights Quantization
        weight_quant = torch.sign(self.weight)

        # MatMul with 1-bit weights using torch.matmul for explicit operation
        output = torch.matmul(input_quant, weight_quant.t())

        # Adding bias if it exists
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)

        # Dequantization with learnable parameters
        output = output * self.beta.unsqueeze(0).expand_as(output)

        return output
    
def replace_linears_in_hf(
    model,
):
    """
    Replaces all instances of nn.Linear in the given model with BitLinear15b.

    Args:
        model (nn.Module): The model to modify.

    Returns:
        None
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace the nn.Linear with BitLinear matching in features and and out_features, and add it to the model
            setattr(
                model,
                name,
                BitLinear15b(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                ),
            )
        else:
            # Recursively apply to child modules
            replace_linears_in_hf(module)


# # Example usage:
# # Load a model from Hugging Face's Transformers
# model_name = "bert-base-uncased"  # Example model
# model = AutoModel.from_pretrained(model_name)

# # Replace its Linear layers with BitLinear
# replace_linears_in_hf(model)

# # Now, `model` has its Linear layers replaced with BitLinear
# print(model)