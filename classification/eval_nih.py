import argparse
import os
import sys
import time
from pathlib import Path
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from classification.utils.configv2 import get_config
from classification.models import build_model
from classification.utils_dino import DINOHead
from timm.utils import AverageMeter
import pandas as pd
import classification.utils_dino as utils_dino

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
torch.set_num_threads(3)
def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #将CuDNN的性能优化模式设置为关闭
    cudnn.benchmark = False
    #将CuDNN的确定性模式设置为启用,确保CuDNN在相同的输入下生成相同的输出
    cudnn.deterministic = True
    #CuDNN加速
    cudnn.enabled = True
    print(cudnn.benchmark, cudnn.deterministic, cudnn.enabled)
def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser = argparse.ArgumentParser('SLGMS', add_help=False)
    # Model parameters
    parser.add_argument('--cfg', type=str,
                        default='../classification/configs/vmambav2v_tiny_224.yaml',
                        metavar="FILE", help='path to config file', )
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils_dino.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.992, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils_dino.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils_dino.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--data-augprob', type=float, default=0., help='the probability that use original input')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--base_lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=2, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.7, 1.),
        help="""(0.4, 1.) Dinoorg Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=9, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.8, 1.),
        help="""(0.05, 0.4) Dinoorg Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--pretrained_pth', type=str, default='.../nih_trainedpt/SLGMS_nih.pth',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')

    # Misc
    parser.add_argument('--output_dir', default='.../nih', type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--self_weight', type=float, default=0.5, help='the probability that use original input')
    parser.add_argument('--cls_weight', type=float, default=1, help='the probability that use original input')
    parser.add_argument('--cls_local_weight', type=float, default=1, help='the probability that use original input')
    return parser

from torchvision import transforms as pth_transforms
from classification.util.chest_dataset import nihchest
def eval_slgms(config, args):
    # utils.init_distributed_mode(args)
    utils_dino.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Grayscale(num_output_channels=3),
        pth_transforms.Resize((448,448)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_test = nihchest(root='.../NIHorg_dataset/',
                            mode='valid',
                            transform=val_transform)

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(config)
    total = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    print("Number of parameters: %.5fM" % (total / 1e6))

    # move networks to gpu
    model = model.cuda()

    state_dict = torch.load(args.pretrained_pth, map_location='cpu')
    state_dict = {k.replace("backbone.", ''): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    validate(test_loader, model)



from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import torch
def calculate_metrics(all_outputs, all_targets_onehot, num_classes=14):
    # 将输出通过 Softmax 激活函数转换为概率
    probabilities = torch.nn.Sigmoid()(all_outputs).cpu().numpy()
    # 将 one-hot 编码的目标转换为类别索引
    all_targets = all_targets_onehot.cpu().numpy().argmax(axis=1)
    auc_scores=[]
    for i in range(num_classes):
        auc = roc_auc_score(all_targets_onehot.cpu().numpy()[:, i], probabilities[:, i])
        auc_scores.append(auc)
    auc_scores1 = np.asarray(auc_scores)
    auc_scores1 = np.mean(auc_scores1[~np.isnan(auc_scores1)])
    return auc_scores, auc_scores1 * 100

@torch.no_grad()
def validate(data_loader, model):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()


    end = time.time()
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    for idx, (images,targets) in enumerate(data_loader):
        images = images.cuda()
        labels = targets.cuda()
        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        pred = model(images)
        # 收集输出和目标
        all_outputs.append(pred)
        all_targets.append(labels)
        loss = nn.BCEWithLogitsLoss()(pred, labels)

        loss_meter.update(loss.item(), labels.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 合并所有输出和目标
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    # 计算整体指标
    # 计算整体指标
    auc_scores, test_auc = calculate_metrics(all_outputs, all_targets)
    cls_names = ['Atelectasis',  'Cardiomegaly',  'Consolidation',  'Edema',  'Effusion',
     'Emphysema',  'Fibrosis',  'Hernia',  'Infiltration',  'Mass',  'Nodule', 'Pleural_Thickening',  'Pneumonia',  'Pneumothorax']
    result = {}

    for ii in range(len(cls_names)):
        result[cls_names[ii]] = auc_scores[ii] * 100
    result["Average AUC"] = test_auc
    df = pd.DataFrame([result]).T  # 使用 .T 进行转置
    print(df)
    df.to_csv(config.OUTPUT + '/test_result_829.csv', index=False)
    # 打印结果
    print(f'AUC: {test_auc:.4f}')
    return loss_meter.avg, test_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SLGMS', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = get_config(args)
    config.defrost()
    config.MODEL.NUM_CLASSES = 14
    config.freeze()
    avg = eval_slgms(config, args)

