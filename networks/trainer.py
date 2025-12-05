import functools
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=False, num_classes=1, adaptive_npr=getattr(opt,'adaptive_npr',False))
            # Optional preload with pretrained weights
            pp = getattr(opt, 'pretrained_path', '')
            if pp:
                try:
                    state = torch.load(pp, map_location='cpu')
                    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
                    from collections import OrderedDict
                    if any(k.startswith('module.') for k in sd.keys()):
                        sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
                    missing, unexpected = self.model.load_state_dict(sd, strict=False)
                    print(f"[init] Loaded pretrained weights from {pp}. Missing:{len(missing)} Unexpected:{len(unexpected)}")
                except Exception as e:
                    print(f"[init] Warning: failed to load pretrained weights from {pp}: {e}")

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1, adaptive_npr=getattr(opt,'adaptive_npr',False))

        if self.isTrain:
            smoothing = getattr(opt, 'label_smoothing', 0.0)
            if smoothing > 0:
                self.loss_fn = lambda logits, targets: nn.functional.binary_cross_entropy_with_logits(
                    logits, targets.clamp(min=smoothing, max=1.0 - smoothing))
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")
            self.use_amp = getattr(opt, 'use_amp', False)
            self.scaler = GradScaler(enabled=self.use_amp)
            if getattr(opt,'cosine_lr', False):
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=opt.niter, eta_min=1e-6)
            else:
                self.scheduler = None

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        
        # Handle CPU-only training (when CUDA not available)
        if torch.cuda.is_available() and len(opt.gpu_ids) > 0:
            self.model.to(opt.gpu_ids[0])
        else:
            self.model.to('cpu')
 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            self.forward()
            self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        if self.use_amp:
            self.scaler.scale(self.loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.loss.backward()
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

