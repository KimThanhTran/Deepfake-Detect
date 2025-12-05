"""
SCRIPT TRAINING MODEL PHÁT HIỆN DEEPFAKE
==========================================

Script này thực hiện training model ResNet-50 với NPR (Noise Print Regularization)
để phát hiện ảnh deepfake.

Chức năng chính:
- Load và preprocess dataset ForenSynths
- Training model với Adam optimizer và cosine learning rate schedule
- Validation định kỳ trên validation set
- Cross-validation trên 8 loại GAN (ProGAN, StyleGAN, etc.)
- Lưu checkpoint và TensorBoard logs

Cách sử dụng:
    python train.py --name my_model --dataroot dataset/ForenSynths --gpu_ids 0
    
Tham số quan trọng:
    --name: Tên experiment (checkpoint sẽ lưu trong checkpoints/<name>/)
    --dataroot: Đường dẫn đến dataset
    --batch_size: Số ảnh xử lý cùng lúc (default: 32)
    --niter: Số epochs training (default: 30)
    --lr: Learning rate ban đầu (default: 0.0001)
"""

import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image

# TensorBoard để visualize training progress
try:
    from tensorboardX import SummaryWriter  # type: ignore[import-not-found]
except Exception:
    try:
        from torch.utils.tensorboard import SummaryWriter  # fallback nếu tensorboardX chưa cài
    except Exception:
        SummaryWriter = None

import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger

import random

def seed_torch(seed=1029):
    """
    Cố định random seed để kết quả reproducible
    
    Args:
        seed (int): Seed value (default: 1029)
        
    Chức năng:
        - Set seed cho Python random, NumPy, PyTorch
        - Đảm bảo kết quả training giống nhau mỗi lần chạy
        - Quan trọng cho research và debugging
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Hash seed cho Python
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch single GPU seed
    torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU seed
    torch.backends.cudnn.benchmark = False  # Tắt auto-tuning (chậm hơn nhưng reproducible)
    torch.backends.cudnn.deterministic = True  # Đảm bảo deterministic algorithms
    torch.backends.cudnn.enabled = False  # Tắt cuDNN để tăng reproducibility


# Config cho cross-validation trên 8 loại GAN
# Danh sách các loại GAN để test
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
# 1 = multiclass (nhiều categories con), 0 = binary (chỉ real/fake)
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(getattr(opt, 'seed', 100))
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)) )
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)

    if SummaryWriter is not None:
        train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
        val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    else:
        train_writer = val_writer = None
    
    model = Trainer(opt)
    
    def testmodel():
        print('*'*25);accs = [];aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.no_resize = False
            Testopt.no_crop = True
            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc);aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    if not getattr(opt, 'skip_bench_eval', False):
        model.eval();testmodel();
    model.train()
    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                if train_writer:
                    train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()
            

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        if val_writer:
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        if not getattr(opt, 'skip_bench_eval', False):
            testmodel()
        model.train()

    if not getattr(opt, 'skip_bench_eval', False):
        model.eval();testmodel()
    model.save_networks('last')
    
