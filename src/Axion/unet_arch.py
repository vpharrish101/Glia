import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self,
                x: torch.Tensor)->torch.Tensor:
        
        return self.block(x)


class UNet(nn.Module):

    def __init__(self,
                in_channels: int=3,
                out_channels: int=1,
                features: list|None=None,):
        
        super().__init__()
        features=features or [32,64,128,256]

        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.encoder_blocks=nn.ModuleList()
        self.decoder_blocks=nn.ModuleList()
        self.upconvs=nn.ModuleList()
        prev=in_channels
        for f in features:
            self.encoder_blocks.append(DoubleConv(prev,f))
            prev=f

        self.bottleneck=DoubleConv(features[-1],features[-1]*2)
        prev=features[-1]*2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev,f,kernel_size=2,stride=2))
            self.decoder_blocks.append(DoubleConv(f*2,f))
            prev=f
        self.final_conv=nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,
                x: torch.Tensor)->torch.Tensor:
        
        skips=[]
        for enc in self.encoder_blocks:
            x=enc(x)
            skips.append(x)
            x=self.pool(x)
        x=self.bottleneck(x)
        for up,dec,skip in zip(self.upconvs,self.decoder_blocks,reversed(skips)):
            x=up(x)
            if x.shape!=skip.shape:
                x=nn.functional.interpolate(x,size=skip.shape[2:],mode="bilinear",align_corners=True)
            x=torch.cat([skip,x],dim=1)
            x=dec(x)

        return self.final_conv(x)


def count_parameters(model: nn.Module)->int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__=="__main__":
    model=UNet(in_channels=3,out_channels=1,features=[32,64,128,256])
    x=torch.randn(1,3,256,256)
    out=model(x)
