import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import sys
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


def calculate_confidence(y_pred):
    num_samp = len(y_pred)
    confidence = torch.zeros(num_samp).cuda()
    # compute difference score for each pair
    for i in range(num_samp):
        for j in range(num_samp):
            # compute class-wise probability difference
            confidence[j] += torch.sum(torch.abs(torch.sub(y_pred[i], y_pred[j])))
    return confidence


def consistencyIndexes(logits, labels, num_classes):
    # get softmax of logits
    y_pred = F.softmax(logits)

    confident_ind, unconfident_ind = [], []
    indexes = np.array([i for i in range(len(labels))])

    for i in range(num_classes):
        # get all indexes that have i as their label
        i_labels = (labels == i).nonzero().flatten().tolist()
        y_pred_i = y_pred[i_labels]
        indexes_i = indexes[i_labels]

        # obtain confidence
        confidence = calculate_confidence(y_pred_i)

        # calculate average confidence
        avg_conf = torch.mean(confidence)

        # print('confidence:', confidence)
        # print('avg:', avg_conf)

        # conf_ind
        confident_ind += indexes_i[((confidence <
                                  avg_conf).nonzero().flatten().tolist())].tolist()
        # unconf_ind
        unconfident_ind += indexes_i[((confidence >=
                                    avg_conf).nonzero().flatten().tolist())].tolist()

    return confident_ind, unconfident_ind
