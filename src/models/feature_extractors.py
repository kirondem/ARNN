import torchvision

def get_feature_extractor_backbone():
    # create vgg16 backbone
    model_vgg16 = torchvision.models.vgg16(pretrained=True).features
    #model_vgg16.out_channels = 512
    for param in model_vgg16.parameters():
        param.requires_grad = False
    return model_vgg16