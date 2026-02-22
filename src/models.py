import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

QUANT_BITS = 8
WEIGHT_INIT_STD = 0.02
INIT_POST_BIN_SCALE = 0.01

def sign_0to1(x):
    return torch.where(x >= 0.0, 1.0, -1.0)

class SignSTE(Function):
    @staticmethod
    def forward(ctx, x):
        return sign_0to1(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class GradualActQuant(nn.Module):
    def __init__(self, n_bits=8, momentum=0.1):
        super().__init__()
        self.n_bits = n_bits
        self.q_max = 2**(n_bits - 1) - 1
        self.register_buffer('running_max', torch.tensor(1.0))
        self.momentum = momentum

    def forward(self, x, p, bin_ratio, eps=1e-6):
        if self.training:
            current_max = x.detach().abs().max()
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * current_max
        
        s = self.running_max / self.q_max
        x_scaled = x / (s + 1e-6)
        x_int = (x_scaled.round() - x_scaled).detach() + x_scaled
        x_clamped = torch.clamp(x_int, -self.q_max, self.q_max)
        x_hard = x_clamped * s

        if self.training and p > 0:
            sprinkle_mask = torch.rand_like(x) < p
            x_sprinkled = torch.where(sprinkle_mask, x_hard, x)
        else:
            x_sprinkled = x

        return (1.0 - bin_ratio) * x_sprinkled + bin_ratio * x_hard

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.current_p = 0.0
        self.current_bin_ratio = 0.0

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.post_bin_scale = nn.Parameter(torch.ones(out_features, 1) * INIT_POST_BIN_SCALE)
        self.final_scale = nn.Parameter(torch.ones(out_features))
        self.final_bias = nn.Parameter(torch.zeros(out_features))
        self.quant = GradualActQuant(n_bits = QUANT_BITS)

        nn.init.trunc_normal_(self.weight, std=WEIGHT_INIT_STD)

    def forward(self, x, p=None, bin_ratio=None):
        p = self.current_p if p is None else p
        bin_ratio = self.current_bin_ratio if bin_ratio is None else bin_ratio

        x = self.quant(x, p, bin_ratio)
        w_sign = SignSTE.apply(self.weight)
        w_hard = w_sign * self.post_bin_scale
        
        sprinkle_mask = torch.rand_like(self.weight) < p
        w_sprinkled = torch.where(sprinkle_mask, w_hard, self.weight)
        w_final = (1.0 - bin_ratio) * w_sprinkled + bin_ratio * w_hard

        return F.linear(x, w_final) * self.final_scale + self.final_bias

class BitConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        
        self.current_p = 0.0
        self.current_bin_ratio = 0.0

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.post_bin_scale = nn.Parameter(torch.ones(out_channels, 1, 1, 1) * INIT_POST_BIN_SCALE)
        self.final_scale = nn.Parameter(torch.ones(out_channels))
        self.final_bias = nn.Parameter(torch.zeros(out_channels))
        self.quant = GradualActQuant(n_bits=QUANT_BITS)
        
        nn.init.trunc_normal_(self.weight, std=WEIGHT_INIT_STD)

    def forward(self, x, p=None, bin_ratio=None):
        p = self.current_p if p is None else p
        bin_ratio = self.current_bin_ratio if bin_ratio is None else bin_ratio

        x = self.quant(x, p, bin_ratio)
        w_sign = SignSTE.apply(self.weight)
        w_hard = w_sign * self.post_bin_scale
        
        sprinkle_mask = torch.rand_like(self.weight) < p
        w_sprinkled = torch.where(sprinkle_mask, w_hard, self.weight)
        w_final = (1.0 - bin_ratio) * w_sprinkled + bin_ratio * w_hard

        y = F.conv2d(x, w_final, bias=None, stride=self.stride, 
                     padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y * self.final_scale.view(1, -1, 1, 1) + self.final_bias.view(1, -1, 1, 1)

class BitModelWrapper(nn.Module):
    def __init__(self, model, use_bin_loss=False):
        super().__init__()
        self.model = model
        self.use_bin_loss = use_bin_loss
        if use_bin_loss:
            self.bin_ratio_param = nn.Parameter(torch.tensor(0.0))

    def update_state(self, module, p, bin_ratio):
        for child in module.children():
            if isinstance(child, (BitLinear, BitConv2d)):
                child.current_p = p
                child.current_bin_ratio = bin_ratio
            self.update_state(child, p, bin_ratio)

    def forward(self, x, p, bin_ratio=None):
        ratio_to_use = self.bin_ratio_param if self.use_bin_loss else bin_ratio
        ratio_to_use = ratio_to_use if ratio_to_use is not None else 0.0
        self.update_state(self.model, p, ratio_to_use)
        return self.model(x)

def convert_to_bit_model(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new_layer = BitConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding, dilation=module.dilation,
                groups=module.groups, bias=(module.bias is not None)
            )
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if module.bias is not None: new_layer.final_bias.copy_(module.bias)
                new_layer.final_scale.fill_(1.0)
            setattr(model, name, new_layer)
        elif isinstance(module, nn.Linear):
            new_layer = BitLinear(
                module.in_features, module.out_features, bias=(module.bias is not None)
            )
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if module.bias is not None: new_layer.final_bias.copy_(module.bias)
                new_layer.final_scale.fill_(1.0)
            setattr(model, name, new_layer)
        else:
            convert_to_bit_model(module)
    return model