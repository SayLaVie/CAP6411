import torch
import torch.nn as nn
import copy
import math
from basisLayer import basisConv2d, basisLinear
ONNX_EXPORT = False
import torch.nn.functional as F


def display_stats(basis_model, model, exp_name, input_size=None):
    if input_size is None:
        input_size = [416, 416]
    print('\n############################################# '+ exp_name +' #############################################\n')
    # print(exp_name)
    num_model_param = sum(p.numel() for p in model.parameters())
    num_basis_param = sum(p.numel() for p in basis_model.parameters())

    org_flops, basis_flops = basis_model.yolo_get_flops(input_size)
    print('    Model FLOPs: %.2fB' % (org_flops / 10**9))
    print('    Basis Model FLOPs: %.2fB' % (basis_flops / 10**9))
    print('    %% Reduction in FLOPs: %.2f %%\n' % (100 - (basis_flops*100/org_flops)))

    print('    Model Params: %.2fM' % (num_model_param / 10**6))
    print('    Basis Model params: %.2fM' % (num_basis_param / 10**6))
    print('    %% Reduction in params: %.2f %%\n' % (100 - (num_basis_param * 100 / num_model_param) ))

    org_filters, basis_filters = sum(basis_model.num_original_filters.tolist()), sum(basis_model.num_basis_filters.tolist())
    print('    Model Filters: %d' % (org_filters))
    print('    Basis Model Filters: %d' % (basis_filters))
    print('    %% Reduction in Filters: %.2f %%\n' % (100 - (basis_filters*100.0/org_filters)))

    print('    Model Accuracy: ')
    print('    Basis Model Accuracy: ')
    print('    Reduction in Accuracy: \n')

    print('    Filter in original convs: ', basis_model.num_original_filters.tolist())
    print('    Filter in original basis convs: ', basis_model.num_basis_filters.tolist())
    print('\n#########################################################################################################\n')

# def display_stats(model, input_size=None):
#     if input_size is None:
#         input_size = [416, 416]
#     print('\n############################################# Network Stats #############################################\n')
#     print('    Total cofficients: %.2fM' % (sum(p.numel() for p in model.basis_parameters()) / 1000000.0))
#     print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
#     org_flops, basis_flops = model.get_flops(input_size)
#     print('    # of multiplications in conv: %.2fB' % (org_flops / 10**9))
#     print('    # of multiplications in basis conv: %.2fB' % (basis_flops / 10**9))
#     print('    %% of multiplications in basis conv: %.1f %%' % (basis_flops*100/org_flops))
#     print('    Filter in original convs: ', model.num_original_filters.tolist())
#     print('    Filter in original basis convs: ', model.num_basis_filters.tolist())
#     print('\n#########################################################################################################\n')

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

# def replace_layer(module, use_weights, add_bn, is_replaced, count=0):
#
#     for n, m in list(module.named_children()):
#         if isinstance(m, nn.Conv2d):
#
#             if is_replaced[count]:
#                 print('Processing conv ', count)
#                 basis_channels = min(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1])
#                 module._modules[n] = basisConv2d(m, basis_channels, use_weights[count], add_bn[count])
#                 ## Verify that the reconstruction is correct
#                 # x = torch.randn(4, m.in_channels, 9, 9)
#                 # o1 = m(x)
#                 # o2 = module._modules[n](x)
#                 #
#                 # print((o1-o2).abs().mean())
#             else:
#                 print('Skipping conv ', count)
#             count += 1
#         elif isinstance(m, nn.Linear):
#
#             if is_replaced[count]:
#                 print('Processing linear ', count)
#                 basis_channels = min(m.out_features, m.in_features)
#                 module._modules[n] = basisLinear(m, basis_channels, use_weights[count], add_bn[count])
#                 ## Verify that the reconstruction is correct
#                 # x = torch.randn(4, m.in_features)
#                 # o1 = m(x)
#                 # o2 = module._modules[n](x)
#                 #
#                 # print((o1-o2).abs().mean())
#             else:
#                 print('Skipping linear ', count)
#             count += 1
#         else:
#             count = replace_layer(m, use_weights, add_bn, is_replaced, count)
#
#     return count

def replace_layer(module, use_weights, add_bn, trainable_basis, replace_fc, count=0):

    for n, m in list(module.named_children()):
        if isinstance(m, nn.Conv2d):

            if replace_fc[count]:
                print('Processing conv ', count)
                basis_channels = min(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1])
                module._modules[n] = basisConv2d(m, basis_channels, use_weights[count], add_bn[count], trainable_basis[count])
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

            if replace_fc[count]:
                print('Processing linear ', count)
                basis_channels = min(m.out_features, m.in_features)
                module._modules[n] = basisLinear(m, basis_channels, use_weights[count], add_bn[count], trainable_basis[count])
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
            count = replace_layer(m, use_weights, add_bn, trainable_basis, replace_fc, count)

    return count

class basisModel(nn.Module):
    def __init__(self, model, use_weights, add_bn, trainable_basis, replace_fc = False):
        super(basisModel, self).__init__()

        self.__dict__ = model.__dict__.copy()

        _, original_channels, basis_channels, layer_type = trace_model(self)
        num_layers = len(basis_channels)

        use_weights = [use_weights] * num_layers
        add_bn = [add_bn] * num_layers
        trainable_basis = [trainable_basis]*num_layers

        if not isinstance(replace_fc, list):
            if replace_fc is False:
                replace_fc = [True] * sum((x == 'conv') for x in layer_type) + [False] * sum((x == 'linear') for x in layer_type)
            elif replace_fc is True:
                replace_fc = [True] * num_layers

        replace_layer(self, use_weights, add_bn, trainable_basis, replace_fc)
        ## Till this point all lists () lenght is equal to num_layers

        ## Remove entries for the layers which are not replaced
        pruned_basis_channels = []
        pruned_original_channels = []
        for i, n in enumerate(replace_fc):
            if n is True:
                pruned_basis_channels.append(basis_channels[i])
                pruned_original_channels.append(original_channels[i])

        ## Save all variables
        self.register_buffer('num_original_filters', torch.IntTensor(pruned_original_channels))
        self.register_buffer('num_basis_filters', torch.IntTensor(pruned_basis_channels))
        self.register_buffer('num_layers', torch.IntTensor([len(pruned_basis_channels)]))


    def cuda(self, device=None):
        super(basisModel, self).cuda(device)
        for m in self.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.cpu()
                m.cuda()

    def cpu(self):
        super(basisModel, self).cpu()

    def update_channels(self, th_T):
        #th_T is either a float value {0 - 1}
        # Or a list of float values {0 - 1}
        # Or a list of Intigers

        # th_T is a float value {0 - 1}
        if not isinstance(th_T, list):
            count = 0
            for m in self.modules():
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
                    for m in self.modules():
                        if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                            c_sum = torch.cumsum(m.delta, 0)
                            c_sum = c_sum / c_sum[-1]
                            idx = torch.nonzero(c_sum >= th_T[count])[0]
                            self.num_basis_filters[count] = torch.IntTensor([idx[0] + 1])
                            m.update_channels(idx[0] + 1)
                            count += 1

                    print('update_channels: Model compression is updated to', th_T)
                ###############################################################################################
                # th_T is a list of Intigers
                elif all(isinstance(x, int) for x in th_T):
                    count = 0
                    self.num_basis_filters = torch.IntTensor(th_T)
                    for m in self.modules():
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
        for m in self.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.update_channels(state_dict['num_basis_filters'][count].item())
                count += 1

        super(basisModel, self).load_state_dict(state_dict, strict)
        print('load_state_dict: Model compression is updated')

    def randomize_filters(self):

        for m in self.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.randomize_filters()

    def basis_parameters(self):
        basis_param = []
        for m in self.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                basis_param.extend(m.parameters())

        return basis_param


class basisYOLO(basisModel):

    def __init__(self, model, use_weights, add_bn, trainable_basis, replace_fc = False):
        model_copy = copy.deepcopy(model)
        super(basisYOLO, self).__init__(model_copy, use_weights, add_bn, trainable_basis, replace_fc = False)
        # self.org_model = model

    def yolo_get_flops(self, input_size):

        info = {'org_flops': [], 'basis_flops': [], 'input_size': [], 'output_size': [], 'layer_name': [],
                'in_channels': [], 'out_channels': [], 'basis_channels': [], 'kernel_size': []}

        for m in self.modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.flops_flag = True

        device = next(self.parameters()).device
        x = torch.rand(1, 3, *input_size).to(device)

        train_mode = self.training
        self.eval()

        self(x)
        self.train(train_mode)

        org_flops = 0
        basis_flops = 0
        for n, m in self.named_modules():
            if isinstance(m, basisConv2d) or isinstance(m, basisLinear):
                m.flops_flag = False
                org_flops += m.flops[0]
                basis_flops += m.flops[1]

                info['org_flops'].append(m.flops[0])
                info['basis_flops'].append(m.flops[1])
                info['input_size'].append(m.flops[2])
                info['output_size'].append(m.flops[3])
                info['layer_name'].append('model.'+n)

                info['in_channels'].append(m.in_channels if isinstance(m, basisConv2d) else m.in_features)
                info['out_channels'].append(m.out_channels if isinstance(m, basisConv2d) else m.out_features)
                info['basis_channels'].append(m.basis_channels.item() if isinstance(m, basisConv2d) else m.basis_features.item())
                info['kernel_size'].append(m.kernel_size if isinstance(m, basisConv2d) else [])

        return org_flops, basis_flops

    def forward(self, x, var=None):
        img_size = x.shape[-2:]
        layer_outputs = []
        output = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            # print(mtype, x.shape)
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    # print(''), [print(layer_outputs[i].shape) for i in layers], print(x.shape)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
            elif mtype == 'yolo':
                x = module(x, img_size)
                output.append(x)
            layer_outputs.append(x if i in self.routs else [])

        if self.training:
            return output
        elif ONNX_EXPORT:
            output = torch.cat(output, 1)  # cat 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            nc = self.module_list[self.yolo_layers[0]].nc  # number of classes
            return output[5:5 + nc].t(), output[:4].t()  # ONNX scores, boxes
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p
