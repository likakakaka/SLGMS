import argparse
import os
import sys
import math
import json
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from classification.data import transforms
from torchvision import models as torchvision_models
from classification.utils.configv2 import get_config
from classification.models import build_model
from classification.utils_dino import DINOHead
from classification.util.datasets_pneum import Shanxi_Mixed_Mask_Dataset_DINO
from classification.util.datasets_pneum import Shanxi_Dataset_DINO as Shanxi_Dataset
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
    parser = argparse.ArgumentParser('SLGMS', add_help=False)
    # Model parameters
    parser.add_argument('--cfg', type=str,
                        default='../classification/configs/vmambav2v_tiny_224.yaml',
                        metavar="FILE", help='path to config file', )
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
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
    parser.add_argument("--base_lr", default=3e-5, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=2, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
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
    parser.add_argument('--pretrained_weights', type=str, default='.../imagenetpt/vssm1_tiny_0230s_ckpt_epoch_264.pth',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    # Misc
    parser.add_argument('--data_path', default='.../shanxi_dataset/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".../result/", type=str, help='Path to save logs and checkpoints.')
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
def train_dino(config, args):
    utils_dino.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset_train = Shanxi_Mixed_Mask_Dataset_DINO(args.data_path + 'segimg_ww_wl224',
                                                   args.data_path + 'segmask_ww_wl224',
                                                   args.data_path + 'train.txt',
                                                   data_transform=transform)

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = Shanxi_Dataset(args.data_path + 'segimg_ww_wl224',
                                 args.data_path + 'val.txt',
                                 data_transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_tasks = utils_dino.get_world_size()
    global_rank = utils_dino.get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank,
                                                  shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset_train)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions

    student = build_model(config)
    teacher = build_model(config)
    embed_dim = student.embed_dim
    total = sum([p.nelement() for p in student.parameters() if p.requires_grad])
    print("Number of parameters: %.5fM" % (total / 1e6))
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils_dino.MultiCropWrapper_new(
        student,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    teacher = utils_dino.MultiCropWrapper_new(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    if utils_dino.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
    state_dict = checkpoint['model']
    del state_dict['classifier.head.weight']
    del state_dict['classifier.head.bias']

    state_dict = {"backbone."+k: v for k, v in state_dict.items()}
    msg = student.load_state_dict(state_dict, strict=False)
    print(msg)
    teacher_without_ddp.load_state_dict(student.state_dict())
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both SLGMS network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils_dino.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    # ============ init schedulers ... ============
    lr_schedule = utils_dino.cosine_scheduler(
        args.base_lr,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils_dino.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils_dino.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "best_avg": 0., "best_epoch": 0}
    start_epoch = to_restore["epoch"]
    best_avg = to_restore["best_avg"]
    best_epoch = to_restore["best_epoch"]
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        acc1,auc,Sensitivity,Specificity, avg = validate_network(val_loader, teacher)
        print(
            f"Avg at epoch {epoch} of the network on the {len(dataset_val)} val images: {avg:.2f}%")

        if avg > best_avg:
            best_avg = avg
            best_epoch=epoch + 1
            # ============ writing logs ... ============
            torch.save(teacher.state_dict(), os.path.join(args.output_dir, "checkpoint_best.pth"))
        # ============ writing logs ... ============
        utils_dino.save_on_master(teacher.state_dict(), os.path.join(args.output_dir, 'checkpoint.pth'))
        print('epoch:', epoch,
              "current avg: {avg:.2f}".format(avg=avg),
              "Val best epoch: {best_epoch:}".format(best_epoch=best_epoch),
              "Val best avg: {avg:.2f}".format(avg=best_avg)
              )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, "current avg: {avg:.2f}": avg,
                     "best_epoch": best_epoch, "val best avg": best_avg}

        if utils_dino.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")



def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils_dino.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils_dino.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, targets,_) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        targets = [target.cuda(non_blocking=True) for target in targets]


        with torch.cuda.amp.autocast(fp16_scaler is not None):
            tea_cls_output, teacher_output, _ = teacher(images[:2], targets, last_fea_return=True,
                                                        last_attn_return=False)  # only the 2 global views pass through the teacher
            stu_cls_output, student_output, targets_output = student(images, targets, last_fea_return=True,
                                                        last_attn_return=False)
            loss, cls_loss, selfloss, cls_loss_local= dino_loss(student_output, teacher_output, stu_cls_output, tea_cls_output, targets_output, epoch)


        if not math.isfinite(cls_loss.item()):
            print("cls_loss is {}, stopping training".format(cls_loss.item()), force=True)
            sys.exit(1)
        if not math.isfinite(selfloss.item()):
            print("selfloss is {}, stopping training".format(selfloss.item()), force=True)
            sys.exit(1)
        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils_dino.clip_gradients(student, args.clip_grad)
            utils_dino.cancel_gradients_last_layer(epoch, student,
                                                   args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils_dino.clip_gradients(student, args.clip_grad)
            utils_dino.cancel_gradients_last_layer(epoch, student,
                                                   args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(cls_loss=cls_loss.item())
        metric_logger.update(selfloss=selfloss.item())
        metric_logger.update(cls_loss_local=cls_loss_local.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import torch
def calculate_metrics(all_outputs, all_targets_onehot, num_classes=2):
    # 将输出通过 Softmax 激活函数转换为概率
    probabilities = torch.softmax(all_outputs, dim=1).cpu().numpy()
    predictions = np.argmax(probabilities, axis=1)  # 选择概率最大的类别作为预测结果
    # 将 one-hot 编码的目标转换为类别索引
    all_targets = all_targets_onehot.cpu().numpy().argmax(axis=1)
    # 计算准确率
    accuracy = accuracy_score(all_targets, predictions)

    # 计算 F1 分数、精确率和召回率
    f1 = f1_score(all_targets, predictions, average='weighted')
    precision = precision_score(all_targets, predictions, average='weighted')
    recall = recall_score(all_targets, predictions, average='weighted')

    # 计算每个类别的敏感性和特异性
    sensitivity = []
    specificity = []
    auc_scores=[]
    for i in range(1):
        auc = roc_auc_score(all_targets_onehot.cpu().numpy()[:, i], probabilities[:, i])
        auc_scores.append(auc)
        true_positive = ((predictions == i) & (all_targets == i)).sum()
        true_negative = ((predictions != i) & (all_targets != i)).sum()
        false_positive = ((predictions == i) & (all_targets != i)).sum()
        false_negative = ((predictions != i) & (all_targets == i)).sum()
        sensitivity.append(
            true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0)
        specificity.append(
            true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0)
    auc_scores = np.asarray(auc_scores)
    auc_scores = np.mean(auc_scores[~np.isnan(auc_scores)])
    sensitivity = np.asarray(sensitivity)
    sensitivity = np.mean(sensitivity[~np.isnan(sensitivity)])
    specificity = np.asarray(specificity)
    specificity = np.mean(specificity[~np.isnan(specificity)])
    return accuracy* 100, auc_scores* 100, sensitivity* 100, specificity* 100, f1* 100, precision* 100, recall* 100
@torch.no_grad()
def validate_network(val_loader, model):
    model.eval()
    metric_logger = utils_dino.MetricLogger(delimiter="  ")
    header = 'Test:'
    # 初始化输出和目标列表
    all_outputs = []
    all_targets = []
    for inp, target, imgname in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp)
        loss = nn.CrossEntropyLoss()(output, target)
        # 收集输出和目标
        all_outputs.append(output)
        all_targets.append(target)
        metric_logger.update(loss=loss.item())
    # 合并所有输出和目标
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    # 计算整体指标
    # 计算整体指标
    accuracy, auc, sensitivity, specificity, f1, precision, recall = calculate_metrics(all_outputs, all_targets)
    avg = (accuracy + auc + sensitivity + specificity) / 4
    # 打印结果
    print(f'AVG: {avg:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, '
          f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, '
          f'F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    metric_logger.meters['avg'].update(avg)
    results = []
    result = {}
    result['Pathology'] = 'AVG'
    result["AUC"] = auc
    result["Acc"] = accuracy
    result["Specificity"] = specificity
    result["Sensitivity"] = sensitivity
    result["avg"] = avg
    result["f1"] = f1
    result["recall"] = recall
    result["precision"] = precision
    results.append(result)
    df = pd.DataFrame(results)
    # print(df)
    df.to_csv(args.output_dir + '/val_result.csv', index=False)
    return accuracy,auc,sensitivity,specificity, avg

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))



    def forward(self, student_output, teacher_output_all, stu_cls_output, tea_cls_output, target, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        kd_loss = 0
        n_loss_terms = 0
        temp = self.teacher_temp_schedule[epoch]
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        teacher_out = F.softmax((teacher_output_all - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        local_global_losses=[]
        # indexes=[]
        for iq in range(2):
            for v, stu in enumerate(student_out):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                q = teacher_out[iq]
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                if v>1:
                    local_global_losses.append(loss.unsqueeze(0))
                    # indexes.append(iq)
                kd_loss += loss.mean()
                n_loss_terms += 1

        kd_loss /= n_loss_terms
        n_cls_loss_terms = 0
        local_global_losses=torch.cat(local_global_losses, dim=0)
        local_global_losses_mean=local_global_losses[:self.ncrops-2]+local_global_losses[self.ncrops-2:]

        target = target.chunk(self.ncrops)
        stu_cls_output = stu_cls_output.chunk(self.ncrops)
        # 局部分区标签为全局标签
        score_mean_global = target[0].detach()
        mask_sorted_sick = torch.zeros_like(local_global_losses_mean)
        for ii in range(local_global_losses_mean.shape[1]):
            label_locali = score_mean_global[ii]
            # 类别1为正常，求概率最小的局部分区值参与分类任务，其余生病求对应类别局部块内概率最大的局部分区参与分类任务
            if torch.argmax(label_locali) != 1:
                # aa = torch.argmax(label_locali)
                cc = local_global_losses_mean[:3, ii]
                # 患病的求局部与全局最相近的分区参与分类任务
                sorted_values, sorted_indices = torch.sort(cc, descending=False)
                top_biggest = sorted_indices[0]
                mask_sorted_sick[top_biggest, ii] = 1

                cc = local_global_losses_mean[3:, ii]
                # 患病的求局部与全局最相近的分区参与分类任务
                sorted_values, sorted_indices = torch.sort(cc, descending=False)
                top_biggest = sorted_indices[0]+3
                mask_sorted_sick[top_biggest, ii] = 1
                second_biggest = sorted_indices[1]+3
                mask_sorted_sick[second_biggest, ii] = 1
            else:
                cc = local_global_losses_mean[:3, ii]
                # 健康的求局部与全局最不相近的倒数第二个分区参与分类任务
                sorted_values, sorted_indices = torch.sort(cc, descending=True)
                # 随机选择 1 或 2
                random_choice = random.choice([1, 2])
                top_biggest = sorted_indices[random_choice]
                mask_sorted_sick[top_biggest, ii] = 1


                cc = local_global_losses_mean[3:, ii]
                # 健康的求局部与全局最不相近的倒数第二个分区参与分类任务
                sorted_values, sorted_indices = torch.sort(cc, descending=True)
                random_choice = random.sample([2, 3,4,5],2)
                third_biggest = sorted_indices[random_choice[0]]+3
                mask_sorted_sick[third_biggest, ii] = 1
                four_biggest = sorted_indices[random_choice[1]]+3
                mask_sorted_sick[four_biggest, ii] = 1


        n_cls_loss_local_allt_terms = 0
        cls_loss_allt_local = torch.tensor(0.).cuda()
        cls_loss = 0

        # teacher centering and sharpening
        for kk in range(self.ncrops - 2):
            mask_local_topone = mask_sorted_sick[kk].detach()
            if torch.any(mask_local_topone == 1.):
                n_cls_loss_local_allt_terms += 1

                cls_loss_allt_local += (nn.CrossEntropyLoss(reduction='none')(stu_cls_output[kk + 2],
                                                                              score_mean_global) * mask_local_topone).sum() / mask_local_topone.sum()

        if n_cls_loss_local_allt_terms != 0:
            cls_loss_allt_local /= n_cls_loss_local_allt_terms

        for v, stu in enumerate(stu_cls_output):
            if v < 2:
                cls_loss += nn.CrossEntropyLoss()(stu, target[v])
                n_cls_loss_terms += 1
        cls_loss /= n_cls_loss_terms
        loss = args.cls_weight * cls_loss + args.cls_local_weight * cls_loss_allt_local + args.self_weight * kd_loss
        self.update_center(teacher_output_all)
        return loss, cls_loss, kd_loss, cls_loss_allt_local

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / len(teacher_output)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomCenterResizedCropHW(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils_dino.random_bbox_maskimgs(0.8),

            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomCenterResizedCropHW(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils_dino.Solarizationimgs(0.2),
            utils_dino.random_bbox_maskimgs(0.2),
            normalize,
        ])

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo_regions = transforms.NineRegionsMixedMaskCenterCrop(96, interpolation=Image.BICUBIC)
        self.local_transfo = transforms.Compose([
            transforms.RandomCenterResizedCropHW(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            # utils_dino.random_bbox_maskimgs(0.5),
            normalize,
        ])

    def __call__(self, image, mask, mixedimg, mixedmask, imgname):
        crops_imgs=[]
        image1, mask1=self.global_transfo1(image, mask)
        crops_imgs.append(image1)
        image2, mask2 = self.global_transfo2(image, mask)
        crops_imgs.append(image2)
        image3,mask3 = self.local_transfo_regions(image, mixedimg, mask, mixedmask)

        for ii in range(len(image3)):
            imga, maska = self.local_transfo(image3[ii], mask3[ii])
            crops_imgs.append(imga)
        return crops_imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SLGMS', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = get_config(args)
    config.defrost()
    config.MODEL.NUM_CLASSES = 2
    config.freeze()
    train_dino(config, args)


