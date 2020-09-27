import torch
import torch.nn as nn
import copy
import math
from .basisLayer import basisConv2d, basisLinear

def display_stats(model, input_size=None):
    if input_size is None:
        input_size = [32, 32]
    print('\n############################################# Network Stats #############################################\n')
    print('    Total cofficients: %.2fM' % (sum(p.numel() for p in model.basis_parameters()) / 1000000.0))
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    num_mul_conv, num_mul_basis, info = model.num_mul(input_size)
    print('    # of multiplications in conv: %.2fB' % (num_mul_conv / 10**9))
    print('    # of multiplications in basis conv: %.2fB' % (num_mul_basis / 10**9))
    print('    %% of multiplications in basis conv: %.1f %%' % (num_mul_basis*100/num_mul_conv))
    print('    Filter in original convs: ', model.num_original_filters.tolist())
    print('    Filter in original basis convs: ', model.num_basis_filters.tolist())
    print('\n#########################################################################################################\n')


def _pair(n):
    if isinstance(n, list) or isinstance(n, tuple):
        return n
    else:
        return (n, n)

def calc_output_size_avgpool(input_size, kernel_size, stride, padding):
    return math.floor( (input_size + 2 * padding - kernel_size) / stride + 1)
    
def get_output_size_avgpool(input_size, kernel_size, stride, padding):

    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)

    a = calc_output_size_avgpool(input_size[0], kernel_size[0], stride[0], padding[0])
    b = calc_output_size_avgpool(input_size[1], kernel_size[1], stride[1], padding[1])

    return [a,b]

def calc_output_size_conv(input_size, kernel_size, stride, padding, dilation):
    return math.floor( (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

def get_output_size_conv(input_size, kernel_size, stride, padding, dilation):

    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    a = calc_output_size_conv(input_size[0], kernel_size[0], stride[0], padding[0], dilation[0])
    b = calc_output_size_conv(input_size[1], kernel_size[1], stride[1], padding[1], dilation[1])

    return [a,b]

def trace_model(model):
    in_channels = []
    out_channels = []
    basis_channels = []
    layer_type = []

    for n, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d):

            in_channels.append(m.in_channels)
            out_channels.append(m.out_channels)
            basis_channels.append( min(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]) )
            layer_type.append('conv')

        elif isinstance(m, nn.Linear):

            in_channels.append(m.in_features)
            out_channels.append(m.out_features)
            basis_channels.append(min(m.out_features, m.in_features))
            layer_type.append('linear')

    return in_channels, out_channels, basis_channels, layer_type

def replace_layer(module, use_weights, add_bn, fixed_basis, is_replaced, count=0):

    for n, m in list(module.named_children()):
        if isinstance(m, nn.Conv2d):

            if is_replaced[count]:
                # print('Processing conv ', count)
                basis_channels = min(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1])
                module._modules[n] = basisConv2d(m, basis_channels, use_weights[count], add_bn[count], fixed_basis[count])
                ## Verify that the reconstruction is correct
                # x = torch.randn(4, m.in_channels, 9, 9)
                # o1 = m(x)
                # o2 = module._modules[n](x)
                #
                # print((o1-o2).abs().mean())
            else:
                print('Skipping conv ', count)
            count += 1
        elif isinstance(m, nn.Linear):

            if is_replaced[count]:
                print('Processing linear ', count)
                basis_channels = min(m.out_features, m.in_features)
                module._modules[n] = basisLinear(m, basis_channels, use_weights[count], add_bn[count])
                ## Verify that the reconstruction is correct
                # x = torch.randn(4, m.in_features)
                # o1 = m(x)
                # o2 = module._modules[n](x)
                #
                # print((o1-o2).abs().mean())
            else:
                print('Skipping linear ', count)
            count += 1
        else:
            count = replace_layer(m, use_weights, add_bn, fixed_basis, is_replaced, count)

    return count

def calc_num_mul_layer(layer_type, in_channels, out_channels, basis_channels, kernel_size=None, output_size=None):

    if layer_type == 1: # basisConv layer
        filter_mul = kernel_size[0] * kernel_size[1] * in_channels
        filter_add = filter_mul
        org_mul_num = output_size[0] * output_size[1] * (filter_mul + filter_add) * out_channels
        bconv_mul_num = output_size[0] * output_size[1] * ((filter_mul + filter_add)) * basis_channels
        proj_mul_num = output_size[0] * output_size[1] * (basis_channels + basis_channels) * out_channels

    elif layer_type == 2: # basisLinear layer
        org_mul_num = 2 * in_channels * out_channels
        bconv_mul_num = 2 * in_channels * basis_channels
        proj_mul_num = 2 * basis_channels * out_channels

    return [org_mul_num, bconv_mul_num+proj_mul_num]

def calc_num_mul_model(model, input_size):

    info = {'num_mul_conv': [], 'num_mul_basis': [], 'input_size': [], 'output_size': [], 'layer_name': [], 'in_channels':[], 'out_channels':[], 'basis_channels':[], 'kernel_size': []}

    for n, module in list(model.named_modules()):

        if isinstance(module, basisConv2d):

            output_size = get_output_size_conv(input_size, module.kernel_size, module.stride, module.padding, module.dilation)
            om, bm = calc_num_mul_layer(1, module.in_channels, module.out_channels, module.basis_channels.item(), module.kernel_size, output_size)

            info['num_mul_conv'].append(om)
            info['num_mul_basis'].append(bm)
            info['input_size'].append(input_size)
            info['output_size'].append(output_size)
            info['layer_name'].append('conv')

            info['in_channels'].append(module.in_channels)
            info['out_channels'].append(module.out_channels)
            info['basis_channels'].append(module.basis_channels.item())
            info['kernel_size'].append(module.kernel_size)

        elif isinstance(module, basisLinear):
            om, bm = calc_num_mul_layer(2, module.in_features, module.out_features, module.basis_features.item())

            info['num_mul_conv'].append(om)
            info['num_mul_basis'].append(bm)
            info['input_size'].append(input_size)
            info['output_size'].append(None)
            info['layer_name'].append('linear')

            info['in_channels'].append(module.in_features)
            info['out_channels'].append(module.out_features)
            info['basis_channels'].append(module.basis_features.item())
            info['kernel_size'].append(None)

        elif isinstance(module, nn.MaxPool2d):
            input_size = get_output_size_conv(input_size, module.kernel_size, module.stride, module.padding, module.dilation)

        elif isinstance(module, nn.AvgPool2d):
            input_size = get_output_size_avgpool(input_size, module.kernel_size, module.stride, module.padding)

    return info

class baseModel(nn.Module):
    def __init__(self, model, use_weights, add_bn, fixed_basis, is_replaced):
        super(baseModel, self).__init__()

        self.model = copy.deepcopy(model)
        _, original_channels, basis_channels, layer_type = trace_model(self.model)
        num_layers = len(basis_channels)

        use_weights = [use_weights] * num_layers
        add_bn = [add_bn] * num_layers
        fixed_basis = [fixed_basis] * num_layers

        if not isinstance(is_replaced, list):
            if is_replaced is False:
                is_replaced = [True] * sum((x == 'conv') for x in layer_type) + [False] * sum((x == 'linear') for x in layer_type)
            elif is_replaced is True:
                is_replaced = [True] * num_layers

        replace_layer(self.model, use_weights, add_bn, fixed_basis, is_replaced)
        ## Till this point all lists () lenght is equal to num_layers

        ## Remove entries for the layers which are not replaced
        pruned_basis_channels = []
        pruned_original_channels = []
        for i, n in enumerate(is_replaced):
            if n is True:
                pruned_basis_channels.append(basis_channels[i])
                pruned_original_channels.append(original_channels[i])

        ## Save all variables
        self.register_buffer('num_original_filters', torch.IntTensor(pruned_original_channels))
        self.register_buffer('num_basis_filters', torch.IntTensor(pruned_basis_channels))
        self.register_buffer('num_layers', torch.IntTensor([len(pruned_basis_channels)]))

    def cuda(self, device=None):
        super(baseModel, self).cuda(device)
        for m in self.model.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.cpu()
                m.cuda()

    def cpu(self):
        super(baseModel, self).cpu()

    def update_channels(self, th_T):
        #th_T is either a float value {0 - 1}
        # Or a list of float values {0 - 1}
        # Or a list of Intigers

        # th_T is a float value {0 - 1}
        if not isinstance(th_T, list):
            count = 0
            for m in self.model.modules():
                if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                    c_sum = torch.cumsum(m.delta, 0)
                    c_sum = c_sum / c_sum[-1]
                    idx = torch.nonzero(c_sum >= th_T)[0]
                    self.num_basis_filters[count] = torch.IntTensor([idx[0] + 1])
                    m.update_channels(idx[0] + 1)

                    count += 1

            print('update_channels: Model compression is updated to', th_T)
        else:
            if len(th_T) == self.num_layers:

                ###############################################################################################
                # th_T is a list of float values {0 - 1}
                if all( (x >= 0.0 and x <= 1.0) for x in th_T):
                    count = 0
                    self.num_basis_filters = torch.IntTensor(th_T)
                    for m in self.model.modules():
                        if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                            c_sum = torch.cumsum(m.delta, 0)
                            c_sum = c_sum / c_sum[-1]
                            idx = torch.nonzero(c_sum >= th_T[count])[0]
                            self.num_basis_filters[count] = torch.IntTensor([idx[0] + 1])
                            m.update_channels(idx[0] + 1)
                            count += 1

                    print('update_channels: Model compression is updated to', th_T)
                ###############################################################################################
                # th_T is list of Intigers
                elif all(isinstance(x, int) for x in th_T):
                    count = 0
                    self.num_basis_filters = torch.IntTensor(th_T)
                    for m in self.model.modules():
                        if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                            m.update_channels(th_T[count])
                            count += 1

                    print('update_channels: Model compression is updated to', th_T)
                else:
                    raise ValueError('update_channels: Cannot mix Floats {0.0 to 1.0 } and Ints')
            else:
                raise ValueError('update_channels: len(th_T) is not equal to num_layers')

    def load_state_dict(self, state_dict, strict=True):

        count = 0
        for m in self.model.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.update_channels(state_dict['num_basis_filters'][count].item())
                count += 1

        super(baseModel, self).load_state_dict(state_dict, strict)
        print('load_state_dict: Model compression is updated')

    def randomize_filters(self):

        for m in self.model.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.randomize_filters()

    def basis_parameters(self):
        basis_param = []
        for m in self.model.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                basis_param.extend(m.parameters())

        return basis_param

class basisModel(baseModel):

    def __init__(self, model, use_weights, add_bn, fixed_basis, is_replaced = False):
        super(basisModel, self).__init__(model, use_weights, add_bn, fixed_basis, is_replaced = is_replaced)

    def num_mul(self, input_size):

        info = calc_num_mul_model(self.model, input_size)
        num_mul_conv = sum(info['num_mul_conv'])
        num_mul_basis = sum(info['num_mul_basis'])

        return num_mul_conv, num_mul_basis, info

    def forward(self, x):
        x = self.model(x)
        return x

# import torchvision.models as models
# m = models.vgg16(pretrained=True)
# em = basisVGG(m, True, False)
# em.update_channels(1.0)
#
# x = torch.randn(1, 3, 224, 224)
# o1 = m(x)
# o2 = em(x)
#
# a, b, c = em.num_mul([224, 224])
# print(a[0]/a[1])

