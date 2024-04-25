import SwinUnet_3D
import torch
from einops import rearrange


def testWindowPartitionAndReverse():
    x = torch.randn((1, 224, 224, 224, 1))

    windows = SwinUnet_3D.window_partition_3D(x, 7)
    print(windows.shape)

    x_hat = SwinUnet_3D.window_reverse_3D(windows, 7, 224, 224, 224)
    print(x_hat.shape)


# testWindowPartitionAndReverse()


def testPatchMergingAndExpand():
    x = torch.randn((3, 96, 56, 56, 56))
    print(x.shape)

    pm = SwinUnet_3D.PatchMerging3D(in_dims=96, downscaling_factor=2, out_dims=192)
    y = pm(x)
    print(y.shape)

    pe = SwinUnet_3D.PatchExpand3D(in_dims=192, up_scaling_factor=2, out_dims=96)
    y_hat = rearrange(y, 'b x y z c -> b c x y z')
    x_hat = pe(y_hat)
    x_hat = rearrange(x_hat, 'b x_s y_s z_s c -> b c x_s y_s z_s')
    print(x_hat.shape)


# testPatchMergingAndExpand()

def testSwinUnet3D():
    #               B,  C, X_S, Y_S, Z_S
    x = torch.randn((1, 3, 224, 224, 160))
    window_size = [i // 32 for i in x.shape[2:]]
    seg = SwinUnet_3D.swinUnet_t_3D(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
                                    window_size=window_size, in_channel=3, num_classes=4
                                    )
    '''
    必需保证： X_S%(x_ws*32) == 0
             Y_S%(y_ws*32) == 0
             Z_S%(z_ws%32) == 0
    一般来说x_ws==y_ws==z_ws,此处不相等主要是因为显存限制，
    对于医学图像，Z轴可能会出现和XY轴尺寸不一致
    '''

    y = seg(x)
    print(y.shape)
    # print(seg)
    # from monai.networks.nets import UNETR, UNet, VNet  # , DynUNet,SegResNet
    # kwargs = {'spatial_dims': 3, 'in_channels': 3, 'out_channels': 2,
    #                       'channels': (32, 64, 128, 256, 512), 'strides': (2, 2, 2, 2)}
    # model = UNet(**kwargs)
    # # print(model)


if __name__ == '__main__':
    testSwinUnet3D()
