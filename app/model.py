import torch
import torch.nn as nn

class UNetAutoencoder(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        # Change input channels from 1 to 16 to match checkpoint
        self.enc1 = self.block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.bottleneck = self.block(16, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.block(32, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        self.classifier = nn.Linear(32 * 14 * 14, num_classes)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        b = self.bottleneck(p1)
        b_flat = b.view(b.size(0), -1)
        class_logits = self.classifier(b_flat)
        u1 = self.up1(b)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        out_img = torch.sigmoid(self.final(d1))
        return out_img, class_logits
