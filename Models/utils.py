# models/utils.py
import torch.nn as nn

def weights_init(m):
    # Only initialize if the module has a weight attribute
    if hasattr(m, 'weight') and m.weight is not None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
