# Written by Yixiao Ge

import copy

import torch.nn as nn
import torch

from ..layers.domain_specific_bn import DSBN
from ..layers.domain_specific_bn import  IBN_MODULE_share_in_affine




layer_size = {'layer1':3, 'layer2':4, 'layer3': 6, 'layer4': 3}


# 15. for dsbn-network ibn+dsbn-share-in-afffine
def convert_ibn_share_in_affine_for_dsbn(model, num_domains=2, target_bn_idx=-1, selected_layer=['layer1', 'layer2', 'layer3', 'layer3']):
    """
    convert all bn layers in the model to domain-specific bn layers
    """

    for _, (child_name, child) in enumerate(model.named_children()):
        if child_name in selected_layer:#{'layer1', 'layer2', 'layer3',}:
            # for name, module in child.named_modules():
            #     print('modules:', name)
            for i in range(0,layer_size[child_name]):
                tmp_model = child[i].bn1
                tmp_layer = child[i].bn1.dsbn[0]

                m = IBN_MODULE_share_in_affine(
                    tmp_layer.num_features,
                    num_domains,
                    target_bn_idx,
                    tmp_layer.eps,
                    tmp_layer.momentum,
                    tmp_layer.weight.requires_grad,
                    tmp_layer.bias.requires_grad,
                )
                m.to(next(tmp_model.parameters()).device)
                weight_dict_in = {}
                weight_dict_bn = {}
                weight_dict = tmp_layer.state_dict()
                weight_dict_in['weight'] = weight_dict['weight'][:int(tmp_layer.num_features/2)]
                weight_dict_in['bias'] = weight_dict['bias'][:int(tmp_layer.num_features/2)]

                weight_dict_bn['weight'] = weight_dict['weight'][int(tmp_layer.num_features / 2):]
                weight_dict_bn['bias'] = weight_dict['bias'][int(tmp_layer.num_features / 2):]
                weight_dict_bn['running_mean'] = weight_dict['running_mean'][int(tmp_layer.num_features / 2):]
                weight_dict_bn['running_var'] = weight_dict['running_var'][int(tmp_layer.num_features / 2):]
                weight_dict_bn['num_batches_tracked'] = weight_dict['num_batches_tracked']
                m.IN.load_state_dict(weight_dict_in)
                for idx in range(num_domains):
                    m.dsbn[idx].load_state_dict(weight_dict_bn)
                    #m.DSBN.dsbn[idx].load_state_dict(weight_dict_bn)
                setattr(child[i], 'bn1', m)

                #setattr(model, child_name, m)
        else:
            # recursive searching
            convert_ibn_share_in_affine_for_dsbn(child, num_domains=num_domains, target_bn_idx=target_bn_idx, selected_layer=selected_layer)



#***************************************
#***************************************
#***************************************
############################========================####################

def convert_dsbn(model, num_domains=2, target_bn_idx=-1):
    """
    convert all bn layers in the model to domain-specific bn layers
    """

    for _, (child_name, child) in enumerate(model.named_children()):
        # if child_name in ['layer1', 'layer2']:
        #     continue
        if isinstance(child, nn.BatchNorm2d):
            # BN2d -> DSBN2d
            m = DSBN(
                child.num_features,
                num_domains,
                nn.BatchNorm2d,
                child.eps,
                child.momentum,
                target_bn_idx,
                child.weight.requires_grad,
                child.bias.requires_grad,
            )
            m.to(next(child.parameters()).device)

            for idx in range(num_domains):
                m.dsbn[idx].load_state_dict(child.state_dict())

            setattr(model, child_name, m)

        elif isinstance(child, nn.BatchNorm1d):
            # BN1d -> DSBN1d
            m = DSBN(
                child.num_features,
                num_domains,
                nn.BatchNorm1d,
                child.eps,
                child.momentum,
                target_bn_idx,
                child.weight.requires_grad,
                child.bias.requires_grad,
            )
            m.to(next(child.parameters()).device)

            for idx in range(num_domains):
                m.dsbn[idx].load_state_dict(child.state_dict())

            setattr(model, child_name, m)

        else:
            # recursive searching
            convert_dsbn(child, num_domains=num_domains, target_bn_idx=target_bn_idx)


def convert_bn(model, target_bn_idx=-1):
    """
    convert all domain-specific bn layers in the model back to normal bn layers
    you need to do convert_sync_bn again after this function, if you use sync bn in the
    model
    """

    for _, (child_name, child) in enumerate(model.named_children()):

        if isinstance(child, DSBN):
            # DSBN 1d/2d -> BN 1d/2d
            m = child.batchnorm_layer(child.num_features, 
                    eps=child.dsbn[target_bn_idx].eps, 
                    momentum=child.dsbn[target_bn_idx].momentum)
            m.weight.requires_grad_(child.weight_requires_grad)
            m.bias.requires_grad_(child.bias_requires_grad)
            m.to(next(child.parameters()).device)

            m.load_state_dict(child.dsbn[target_bn_idx].state_dict())

            setattr(model, child_name, m)

        else:
            # recursive searching
            convert_bn(child, target_bn_idx=target_bn_idx)


def extract_single_bn_model(model, target_bn_idx=-1):
    """
    extract a model with normal bn layers from the domain-specific bn models
    """
    model_cp = copy.deepcopy(model)
    convert_bn(model_cp, target_bn_idx=target_bn_idx)
    return model_cp


def switch_target_bn(model, target_bn_idx=-1):
    """
    switch the target_bn_idx of all domain-specific bn layers
    """

    for _, child in model.named_children():

        if isinstance(child, DSBN) or isinstance(child, IBN_MODULE_share_in_affine): #IBN_F_ada_global
            child.target_bn_idx = target_bn_idx

        else:
            # recursive searching
            switch_target_bn(child, target_bn_idx=target_bn_idx)


def convert_bn_new(model, target_bn_idx=-1):
    """
    convert all domain-specific bn layers in the model back to normal bn layers
    you need to do convert_sync_bn again after this function, if you use sync bn in the
    model
    """

    for _, (child_name, child) in enumerate(model.named_children()):

        if isinstance(child, DSBN):
            # DSBN 1d/2d -> BN 1d/2d
            m = child.batchnorm_layer(child.num_features,
                    eps=child.dsbn[target_bn_idx].eps,
                    momentum=child.dsbn[target_bn_idx].momentum)
            m.weight.requires_grad_(child.dsbn[target_bn_idx].weight.requires_grad)
            m.bias.requires_grad_(child.dsbn[target_bn_idx].bias.requires_grad)
            m.to(next(child.parameters()).device)

            m.load_state_dict(child.dsbn[target_bn_idx].state_dict())

            setattr(model, child_name, m)

        else:
            # recursive searching
            convert_bn_new(child, target_bn_idx=target_bn_idx)

