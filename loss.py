import torch
import torch.nn as nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10 # bbox를 더 중요하게 여김

    def forward(self, prediction, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No object loss
        no_object_loss = self.bce(
            (prediction[..., 0:1][noobj], (target[...,0:1][noobj]))
        )

        # Object Loss 
        # prediction = []
        anchors = anchors.reshape(1, 3, 1, 1, 2) # 3 x 2 -> p_2 * exp(t_w)
        box_preds = torch.cat([self.sigmoid(prediction[..., 1:3]), torch.exp(prediction[..., 3:5] * anchors)], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((prediction[..., 0:1][obj]), (ious * target[..., 0, 1]))

        # Box Coordinate Loss
        

        # Class Loss