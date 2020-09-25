# Created by Hojun Lim(Media Informatics, 405165) at 23.09.20

import sys
from tqdm import tqdm
import argparse
import os
import os.path as osp
import pprint
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils import data
from advent.model.deeplabv2 import get_deeplab_v2
from advent.model.discriminator import get_fc_discriminator
from advent.dataset.cityscapes import CityscapesDataSet
from advent.utils.func import prob_2_entropy
import torch.nn.functional as F
from advent.utils.func import loss_calc, bce_loss
from advent.domain_adaptation.config import cfg, cfg_from_file
from matplotlib import pyplot as plt
from matplotlib import image  as mpimg
from entropy_rank.entropy import easy_split_txt_dir, hard_split_txt_dir, save_dir, img_save_dir, color_img_save_dir, \
    colorize, colorize_save, cluster_subdomain, find_rare_class, load_checkpoint_for_evaluation
from entropy_rank.entropy_THRESH import colorize_save_with_thresholding
thresholding = True

# ------------------------------------- color -------------------------------------------
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
# The rare classes trainID from cityscapes dataset
# These classes are:
#    wall, fence, pole, traffic light, trafflic sign, terrain, rider, truck, bus, train, motor.
rare_class = [3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 17]

"""
def colorize_save_with_thresholding(easy_split, thres, predicted_label, predicted_prob, image_name, args):
    for index in range(len(easy_split)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]

        # apply thresholding
        for i in range(cfg.NUM_CLASSES):
            label[(prob < thres[i]) * (
                        label == i)] = 255  # set pixels whose predicted label is 'i' and its confidence score is below then 'thres[i]' as 255
        output_thres = np.asarray(label, dtype=np.uint8)

        mask_Img = Image.fromarray(output_thres)
        mask_color = colorize(output_thres)

        name = name.split('/')[-1]
        # LB = args.LB.replace('.','_')
        mask_Img.save(img_save_dir % (args.FDA_mode, args.LB, str(thresholding), args.round, name))
        mask_color.save(
            color_img_save_dir % (args.FDA_mode, args.LB, str(thresholding), args.round, name.split('.')[0]))
"""


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")

    parser.add_argument('--best_iter', type=int, default=70000,
                        help='iteration with best mIoU')
    parser.add_argument('--normalize', type=bool, default=False,
                        help='add normalizor to the entropy ranking')
    parser.add_argument('--lambda1', type=float, default=0.67,
                        help='hyperparameter lambda to split the target domain')
    parser.add_argument('--cfg', type=str, default='../ADVENT/advent/scripts/configs/advent.yml',
                        help='optional config file')
    # ----------------------------------------------------------------#
    parser.add_argument("--FDA-mode", type=str, default="off",
                        help="on: apply the amplitude switch between source and target, off: doesn't apply ampltude switch")
    parser.add_argument('--round', type=int, default=0, help='specify the round of self supervised learning')
    parser.add_argument("--MBT", type=bool, default=True)
    parser.add_argument("--LB", type=str, default="0", help="beta for FDA or MBT")
    parser.add_argument("--restore-opt1", type=str, default=None, help="restore model parameters from beta1")
    parser.add_argument("--restore-opt2", type=str, default=None, help="restore model parameters from beta2")
    parser.add_argument("--restore-opt3", type=str, default=None, help="restore model parameters from beta3")
    # ----------------------------------------------------------------#
    return parser.parse_args()


def main(args):
    # load configuration file
    device = cfg.GPU_ID
    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)

    if not os.path.exists(save_dir % (args.FDA_mode, args.round)):
        os.mkdir(save_dir % (args.FDA_mode, args.round))
    # ----------------------------------------------------------------#
    args.LB = str(args.MBT) # set args.LB = 'MBT'
    SRC_IMG_MEAN = np.asarray(cfg.TRAIN.IMG_MEAN, dtype=np.float32)
    SRC_IMG_MEAN = torch.reshape(torch.from_numpy(SRC_IMG_MEAN), (1, 3, 1, 1))


    ###################### here, replace by restoring three different model#####################
    if args.round == 0:  # first round of SSL
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}_{args.FDA_mode}_LB_{args.LB}'

    elif args.round > 0:  # when SSL round is higher than 0

        # SOURCE and TARGET are no longer GTA and Cityscape, but are easy and hard split
        cfg.SOURCE = 'CityscapesEasy'
        cfg.TARGET = 'CityscapesHard'
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}_{args.FDA_mode}_LB_{args.LB}_THRESH_{str(thresholding)}_ROUND_{args.round - 1}'
    else:
        raise KeyError()

    ##########################################################################################################################################
    # ----------------------------------------------------------------#
    cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)

    # load model with parameters trained from Inter-domain adaptation
    model_gen = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TEST.MULTI_LEVEL)

    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{args.best_iter}.pth')

    print("Loading the generator:", restore_from)

    load_checkpoint_for_evaluation(model_gen, restore_from, device)

    # load data
    target_dataset = CityscapesDataSet(args=args, root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=None,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)

    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=None)

    target_loader_iter = enumerate(target_loader)

    # upsampling layer
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # ---------------------------------------------------------------------------------------------------------------#

    # step 1. entropy-ranking: split the target dataset into easy and hard cases.

    entropy_list = []
    for index in tqdm(range(len(target_loader))):
        _, batch = target_loader_iter.__next__()
        image, _, _, name = batch

        # normalize the image before fed into the trained model
        B, C, H, W = image.shape
        mean_image = SRC_IMG_MEAN.repeat(B, 1, H, W)

        if args.FDA_mode == 'on':
            image -= mean_image

        elif args.FDA_mode == 'off':
            # no need to perform normalization again since that has been done already in dataset class(GTA5, cityscapes) when args.FDA_mode = 'off'
            image = image

        else:
            raise KeyError()

        with torch.no_grad():
            _, pred_trg_main = model_gen(image.cuda(device))  # shape(pred_trg_main) = (1, 19, 65, 129)
            pred_trg_main = interp_target(pred_trg_main)  # shape(pred_trg_main) = (1, 19, 512, 1024)
            if args.normalize == True:
                normalizor = (11 - len(find_rare_class(pred_trg_main))) / 11.0 + 0.5
            else:
                normalizor = 1
            pred_trg_entropy = prob_2_entropy(F.softmax(pred_trg_main))
            entropy_list.append((name[0], pred_trg_entropy.mean().item() * normalizor))
            # colorize_save(pred_trg_main, name[0], args.FDA_mode)

    # split the enntropy_list into
    _, easy_split = cluster_subdomain(entropy_list, args, thresholding)

    # ---------------------------------------------------------------------------------------------------------------#

    # step2. apply thresholding(either top 66% or confidence score above 0.9) to easy-split target dataset and save them.

    predicted_label = np.zeros((len(easy_split), 512, 1024))  # (512, 1024) is the size of target output
    predicted_prob = np.zeros((len(easy_split), 512, 1024))
    image_name = []
    idx = 0

    target_loader_iter = enumerate(target_loader)

    for index in tqdm(range(len(target_loader))):
        _, batch = target_loader_iter.__next__()
        image, _, _, name = batch

        if name[0] not in easy_split:  # only compute the images that belongs to easy-split
            continue

        # normalize the image before fed into the trained model
        B, C, H, W = image.shape
        mean_image = SRC_IMG_MEAN.repeat(B, 1, H, W)

        if args.FDA_mode == 'on':
            image -= mean_image

        elif args.FDA_mode == 'off':
            # no need to perform normalization again since that has been done already in dataset class(GTA5, cityscapes) when args.FDA_mode = 'off'
            image = image

        else:
            raise KeyError()

        with torch.no_grad():
            _, pred_trg_main = model_gen(image.cuda(device))  # shape(pred_trg_main) = (1, 19, 65, 129)
            pred_trg_main = F.softmax(interp_target(pred_trg_main), dim=1).cpu().data[
                0].numpy()  # shape(pred_trg_main) = (1, 19, 512, 1024)
            pred_trg_main = pred_trg_main.transpose(1, 2, 0)  # shape(pred_trg_main) = (512, 1024, 19)
            label, prob = np.argmax(pred_trg_main, axis=2), np.max(pred_trg_main, axis=2)
            predicted_label[idx] = label
            predicted_prob[idx] = prob
            image_name.append(name[0])
            idx += 1

    assert len(easy_split) == len(image_name)  # check whether all images in easy-split are processed

    # compute the threshold for each label
    thres = []
    for i in range(cfg.NUM_CLASSES):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        thres.append(x[np.int(np.round(
            len(x) * 0.66))])  # thres contains the thresholding values by labels in corresponding entry:thres[label]
    print(thres)
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9

    print(thres)
    colorize_save_with_thresholding(easy_split, thres, predicted_label, predicted_prob, image_name, args)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    main(args)

# command
# python entropy.py --best_iter 62000 --normalize False --lambda1 0.67 --FDA-mode 'on