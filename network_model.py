# 必要なモジュールのインポート
from torchvision import transforms
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

# 前処理
transform = transforms.Compose([
    transforms.ToTensor()
])

class Net(nn.Module):

    def __init__(self, n_feature=1024, n_class=4):
        super().__init__()

        # 以前の初期化方法
        # self.faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)

        # 新しい初期化方法
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.faster_rcnn = fasterrcnn_resnet50_fpn(weights=weights)

        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(n_feature, n_class)


    def forward(self, x, t=None):
        if self.training:
            return self.faster_rcnn(x, t)
        else:
            return self.faster_rcnn(x)