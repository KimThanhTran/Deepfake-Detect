import sys
import os
import torch


def load_npr_state_dict(path):
    """Load an NPR checkpoint into a plain state dict.

    Handles both raw state dicts and {'model': ...} wrappers, and strips the
    'module.' prefix left by DataParallel training.
    """
    state = torch.load(path, map_location='cpu')
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    return sd


def build_npr_model(path, num_classes=1):
    """Build the correct NPR model for a checkpoint and load it strictly.

    Checkpoints trained with --adaptive_npr contain 'npr.*' keys; loading them
    into a fixed-NPR model with strict=False silently drops those weights and
    evaluates the wrong architecture. Detect and construct accordingly.
    """
    from networks.resnet import resnet50
    sd = load_npr_state_dict(path)
    adaptive = any(k.startswith('npr.') for k in sd.keys())
    model = resnet50(num_classes=num_classes, adaptive_npr=adaptive)
    model.load_state_dict(sd)  # strict: fail loudly on any mismatch
    return model, adaptive


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]




class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        
        
def printSet(set_str):
    set_str = str(set_str)
    num = len(set_str)
    print("="*num*3)
    print(" "*num + set_str)
    print("="*num*3)