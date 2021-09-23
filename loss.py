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
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No object loss
        no_object_loss = self.bce(
            (prediction[..., 0:1][noobj]), (target[...,0:1][noobj])
        )

        # Object Loss 
        # prediction = [p_0, x, y, w, h, p_c]
        anchors = anchors.reshape(1, 3, 1, 1, 2) # 3 x 2 -> p_2 * exp(t_w)
        box_preds = torch.cat([self.sigmoid(prediction[..., 1:3]), torch.exp(prediction[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(prediction[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # Box Coordinate Loss
        prediction[..., 1:3] = self.sigmoid(prediction[..., 1:3])
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(prediction[..., 1:5][obj], target[..., 1:5][obj])

        # Class Loss
        # 왜 long 형으로 변환?
        class_loss = self.entropy(
            (prediction[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        # multi loss 구현
        return(
            self.lambda_box * box_loss 
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )