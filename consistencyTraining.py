# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import sys
import time
from consistencySumMetric import consistencyIndexes as sum_metric
import numpy as np
import torchvision.transforms as transforms
# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--noise_type', type=str,
                    help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type=str,
                    help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type=str,
                    help=' cifar10 or cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4,
                    help='how many subprocesses to use for data loading')
parser.add_argument('--is_human', action='store_true', default=False)

# store starting time
begin = time.time()

# Adjust learning rate and for SGD Optimizer


def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train(epoch, train_loader, model, optimizer, num_classes, noise_or_not, train_dataset):
    train_total = 0
    train_correct = 0

    sum_conf_inc = 0
    sum_num_conf = 0
    sum_unconf_inc = 0
    sum_num_unconf = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()

        print('indexes:', indexes)
        print('ind:', ind)
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits = model(images)

        # obtain confidence indexes for sum metric
        sum_confident_ind, sum_unconfident_ind = sum_metric(
            logits, labels, num_classes)

        # calculate how accurate
        sum_confident_samples = ind[sum_confident_ind]
        sum_unconfident_samples = ind[sum_unconfident_ind]

        for index in sum_confident_samples:
            sum_conf_inc += noise_or_not[index]
        sum_num_conf += len(sum_confident_samples)

        for index in sum_unconfident_samples:
            sum_unconf_inc += noise_or_not[index]
        sum_num_unconf += len(sum_unconfident_samples)

        print('conf ind:', sum_confident_ind)
        print('unconf ind:', sum_unconfident_ind)

        # split into confident and unconfident logits and labels
        labels_conf = labels[sum_confident_ind]
        images_conf = images[sum_confident_ind]

        # apply MixUp to confident labels
        conf_inputs, conf_targets_a, conf_targets_b, lam = mixup_data(
            images_conf, labels_conf)
        conf_inputs, conf_targets_a, conf_targets_b = map(
            Variable, (conf_inputs, conf_targets_a, conf_targets_b))

        logits_conf = model(conf_inputs)
        # conf loss
        conf_loss = lam * F.cross_entropy(logits_conf, conf_targets_a, reduce=True) + (
            1 - lam) * F.cross_entropy(logits_conf, conf_targets_b, reduce=True)

        # get unconf
        logits_unconf = logits[sum_unconfident_ind]
        labels_unconf = labels[sum_unconfident_ind]
        # create pseudolabels based on logits on lightly augmented images for unconfident set
        unconf_pseudolabels = torch.argmax(logits_unconf, dim=1)
        print('logits unconf:', logits_unconf)
        print('orig labels:', labels_unconf)
        print('pseudo:', unconf_pseudolabels)

        # heavily augment images
        print('unconf act ind:', sum_unconfident_samples)
        aug_images, aug_lab, aug_ind = train_dataset.getItemRandAug(
            sum_unconfident_samples)
        aug_images = Variable(aug_images).cuda()

        print('images unconf 1:', aug_images)
        print('images unconf 2:', images[sum_unconfident_ind])

        # predict on heavily augmented
        aug_logits = model(aug_images)
        # unconf loss
        unconf_loss = F.cross_entropy(
            aug_logits, unconf_pseudolabels, reduce=True)

        # training accuracy
        prec_a, _ = accuracy(logits_conf, conf_targets_a, topk=(1, 5))
        prec_b, _ = accuracy(logits_conf, conf_targets_b, topk=(1, 5))
        prec_u = accuracy(aug_logits, unconf_pseudolabels, topk=(1, 5))
        prec = lam * prec_a + (1-lam)*prec_b + prec_u
        # prec = 0.0
        train_total += 1
        train_correct += prec
        loss = conf_loss + unconf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))

    print(f'Sum confident noise: {sum_conf_inc} out of {sum_num_conf}')
    print(f'Sum unconfident noise: {sum_unconf_inc} out of {sum_num_unconf}')

    train_acc = float(train_correct)/float(train_total)
    return train_acc, sum_conf_inc, sum_num_conf, sum_unconf_inc, sum_num_unconf
# test
# Evaluate the Model


def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    # print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc


#####################################main code ################################################
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                  'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else:
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
    args.dataset, args.noise_type, args.noise_path, args.is_human)

noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels),
      train_dataset.train_labels[:10])
# load model
print('building model...')
model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           num_workers=args.num_workers,
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          num_workers=args.num_workers,
                                          shuffle=False)
alpha_plan = [0.1] * 60 + [0.01] * 40
model.cuda()


epoch = 0
train_acc = 0

# training
file = open('./checkpoint/%s_%s' %
            (args.dataset, args.noise_type)+'_all.txt', "w")

max_test = 0

for epoch in range(args.n_epoch):
    # train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    train_acc, sum_conf_inc, sum_num_conf, sum_unconf_inc, sum_num_unconf = train(
        epoch, train_loader, model, optimizer, num_classes, noise_or_not, train_dataset)

    # evaluate models
    test_acc = evaluate(test_loader=test_loader, model=model)
    if test_acc > max_test:
        max_test = test_acc
    # save results
    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)
    file.write("\nepoch: "+str(epoch))
    file.write("\n\ttrain acc on train images is "+str(train_acc)+"\n")
    file.write("\ttest acc on test images is "+str(test_acc)+"\n")

    file.write("\t\tSum num of noisy samples in confident: " +
               str(sum_conf_inc)+" out of: " + str(sum_num_conf)+"\n")
    file.write("\t\tSum num of noisy samples in unconfident: " +
               str(sum_unconf_inc)+" out of: " + str(sum_num_unconf)+"\n")

    file.write("\ttest acc on test images is "+str(test_acc)+"\n")
file.write("\n\nfinal test acc on test images is "+str(test_acc)+"\n")
file.write("max test acc on test images is "+str(max_test)+"\n")

# store end time
end = time.time()
timeTaken = time.strftime("%H:%M:%S", time.gmtime(end-begin))
# total time taken
file.write('Total runtime of the program is: ' + timeTaken)
file.flush()
file.close()
