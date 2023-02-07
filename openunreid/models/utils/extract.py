# Written by Yixiao Ge

import time
from collections import OrderedDict

import torch
import torch.nn.functional as F

from ...utils.dist_utils import all_gather_tensor, get_dist_info, synchronize
from ...utils.meters import Meters
import torch.nn as nn


@torch.no_grad()
def extract_features(
    model,  # model used for extracting
    data_loader,  # loading data
    dataset,  # dataset with file paths, etc
    cuda=True,  # extract on GPU
    normalize=True,  # normalize feature
    with_path=False,  # return a dict {path:feat} if True, otherwise, return only feat (Tensor)  # noqa
    print_freq=10,  # log print frequence
    save_memory=False,  # gather features from different GPUs all together or in sequence, only for distributed  # noqa
    for_testing=True,
    prefix="Extract: ",
    for_pseudo=False,
):

    progress = Meters({"Time": ":.3f", "Data": ":.3f"}, len(data_loader), prefix=prefix)

    rank, world_size, is_dist = get_dist_info()
    features = []
    true_label = []

    model.eval()
    ###########################
    # if for_pseudo == True:
    #     convert_bn_test(model, running_stats=False)
    #     #show_bn_test(model)
    ################################

    data_iter = iter(data_loader)

    end = time.time()
    for i in range(len(data_loader)):
        data = next(data_iter)
        progress.update({"Data": time.time() - end})

        images = data["img"]
        tmp_label = data["id"]
        if cuda:
            images = images.cuda()
            tmp_label = tmp_label.cuda()
        # compute output
        outputs = model(images)



        if isinstance(outputs, list) and for_testing:
            outputs = torch.cat(outputs, dim=1)

        if normalize:
            if isinstance(outputs, list):
                outputs = [F.normalize(out, p=2, dim=-1) for out in outputs]
            outputs = F.normalize(outputs, p=2, dim=-1)

        if isinstance(outputs, list):
            outputs = torch.cat(outputs, dim=1).data.cpu()
        else:
            outputs = outputs.data.cpu()

        features.append(outputs)
        true_label.append(tmp_label)

        # measure elapsed time
        progress.update({"Time": time.time() - end})
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    ###########################
    # if for_pseudo == True:
    #     convert_bn_test(model, running_stats=True)
    #     # show_bn_test(model)
    ################################

    synchronize()

    if is_dist and cuda:
        # distributed: gather features from all GPUs
        features = torch.cat(features)
        all_features = all_gather_tensor(features.cuda(), save_memory=save_memory)
        all_features = all_features.cpu()[: len(dataset)]
        ###label
        true_label = torch.cat(true_label)
        all_true_label = all_gather_tensor(true_label.cuda(), save_memory=save_memory)
        all_true_label = all_true_label.cpu()[: len(dataset)]
    else:
        # no distributed, no gather
        all_features = torch.cat(features, dim=0)[: len(dataset)]
        all_true_label = torch.cat(true_label, dim=0)[: len(dataset)]

    if not with_path:
        if prefix == "Cluster: ":
            return all_features, all_true_label
        return all_features

    features_dict = OrderedDict()
    for fname, feat in zip(dataset, all_features):
        features_dict[fname[0]] = feat

    return features_dict


def convert_bn_test(model, running_stats=False):
    for _, (child_name, child) in enumerate(model.named_children()):
        # if child_name in ['layer1', 'layer2']:
        #     continue
        # print("******************************************************")
        # print(child)
        # print("******************************************************")
        if isinstance(child, nn.BatchNorm2d):
            # BN2d -> DSBN2d
            child.track_running_stats=running_stats
            child.training = True
        elif isinstance(child, nn.BatchNorm1d):
            # BN1d -> DSBN1d
            child.track_running_stats = running_stats
            child.training = True
        else:
            # recursive searching
            convert_bn_test(child, running_stats)

       # return model

def show_bn_test(model):
    for _, (child_name, child) in enumerate(model.named_children()):
        # if child_name in ['layer1', 'layer2']:
        #     continue
        print("******************************************************")
        print(child)
        print("******************************************************")
        if isinstance(child, nn.BatchNorm2d):
            # BN2d -> DSBN2d
            print("******************************************************")



        elif isinstance(child, nn.BatchNorm1d):
            # BN1d -> DSBN1d
            #child.track_running_stats = False
            print("******************************************************")

        else:
            # recursive searching
            convert_bn_test(child)
