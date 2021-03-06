#--------------------------------------------------------------------
# modified from "ADVENT/advent/scripts/test.py" by Tuan-Hung Vu
#--------------------------------------------------------------------
# --------------------------------------------------------
# Adaptation of FDA to Intra DA
#
#
# Updated by Hojun Lim
# Update date 15.09.2020
# --------------------------------------------------------

import argparse
import os
import os.path as osp
import pprint
import warnings

from torch.utils import data

from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.eval_UDA import evaluate_domain_adaptation

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    # ----------------------------------------------------------------#
    parser.add_argument("--FDA-mode", type=str, default="off",
                        help="on: apply the amplitude switch between source and target, off: doesn't apply amplitude switch")
    parser.add_argument("--LB", type=float, default=0, help="beta for FDA")
    parser.add_argument("--thres", type=bool, default=False, help="thresholding in entropy-ranking")
    parser.add_argument('--round', type=int, default=0, help='specify the round of self supervised learning')
    parser.add_argument("--MBT", type=bool, default=False)
    # ----------------------------------------------------------------#
    return parser.parse_args()




    # ----------------------------------------------------------------#
def main():

    # LOAD ARGS
    args = get_arguments()
    config_file = args.cfg
    exp_suffix = args.exp_suffix
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)
    # auto-generate exp name if not specified
    # pdb.set_trace()


    if cfg.EXP_NAME == '':
        if args.MBT: # when to train a model on pseudo label from MBT
            cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}_{args.FDA_mode}_LB_MBT_THRESH_{args.thres}_ROUND_{args.round}'
        else:
            args.LB = str(args.LB).replace('.', '_')
            cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}_{args.FDA_mode}_LB_{args.LB}_THRESH_{args.thres}_ROUND_{args.round}'
    # ----------------------------------------------------------------#
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'
    # auto-generate snapshot path if not specified
    # pdb.set_trace()
    if cfg.TEST.SNAPSHOT_DIR[0] == '':
        cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TEST.SNAPSHOT_DIR[0], exist_ok=True)

    print('Using config:')
    pprint.pprint(cfg)
    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == 'best':
        assert n_models == 1, 'Not yet supported'
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == 'DeepLabv2':
            model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                                   multi_level=cfg.TEST.MULTI_LEVEL[i])
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # dataloaders
    # pdb.set_trace()
    # ----------------------------------------------------------------#
    test_dataset = CityscapesDataSet(args= args, root=cfg.DATA_DIRECTORY_TARGET,
                                     list_path='../ADVENT/advent/dataset/cityscapes_list/{}.txt',
                                     set=cfg.TEST.SET_TARGET,
                                     info_path=cfg.TEST.INFO_TARGET,
                                     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                     mean=cfg.TEST.IMG_MEAN,
                                     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE_TARGET,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=False,
                                  pin_memory=True)
    # eval
    # pdb.set_trace()
    evaluate_domain_adaptation(models, test_loader, cfg)
    # ----------------------------------------------------------------#


if __name__ == '__main__':
    #args = get_arguments()
    #print('Called with args:')
    #print(args)
    # ----------------------------------------------------------------#
    #main(args.cfg, args.FDA_mode, args.exp_suffix)
    main()
    # ----------------------------------------------------------------#


   # command
   # python test.py --cfg ./intradaFDA_on --FDA-mode='on'