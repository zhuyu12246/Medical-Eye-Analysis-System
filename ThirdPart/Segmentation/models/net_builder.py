from models.nets.CAMWNet import CAMWNet

def net_builder(name,pretrained_model=None,pretrained=False):
    if name == 'camwnet':
        net = CAMWNet(in_channels = 3,n_classes=6,feature_scale=2)
    else:
        raise NameError("Unknow Model Name!")
    return net
