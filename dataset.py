#import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    iou_width_height as iou,
    non_max_suppression as nms,
)

# 잘린 이미지를 띄우거나, np.array로 변환시 에러가 발생하는데, 이를 해결
ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self, 
        csv_file,
        img_dir, label_dir,
        anchors, # [3, 13, 13], [3, 26, 26], [3, 52, 52]
        image_size=416,
        S=[13, 26, 52], # anchor box의 degree size
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3 
        self.C = C
        self.ignore_iou_thresh = 0.5 # IOU가 0.5보다 크면 무시

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # 공백 기준으로 나눔 + 최소 2차원 array로 반환 
        # np.roll : 첫번째 원소를 4칸 밀고 나머지를 앞으로 끌어옴  
        # (class, x, y, w, h) -> ( x, y, w, h, class)
        # 즉 label값을 0번째에서 4번째로 이동
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndim=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB")) # type을 array로 변경해야된다.

        if self.transform:
            # augmentation 라이브러리는 torch vision에서 사용할 수 없다.
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        # [p_o, x, y, w, h, c]
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # 1개의 GT box와 9개의 anchor 간의 w,h iou
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) #내림차순으로 정렬 
            x, y, width, height, class_label = box
            has_anchor = [False, False, False] # bboxes가 있는지 확인

            for anchor_idx in anchor_indices: #GT와 iou가 큰 anchor 부터
                scale_idx = anchor_idx // self.num_anchors_per_scale # 0, 1, 8->2 52x52를 의미
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # 0, 1, 2
                S = self.S[scale_idx]
                # txt 파일에서 bbox가 Normalize 되었기 때문에 곱셈 수행.
                i, j = int(S*y), int(S*x) # x = 0.5, S = 13 --> int(6.5) -> 6
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # [anchor_idx, 행(y), 열(x), object probability]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S*x - j, S*y - i # 6.5 - 6 = 0.5, both are between [0, 1]
                    width_cell, height_cell = (
                        width * S, # S = 13, width=0.5, 6.5
                        height * S, 
                    )
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # 예측을 실패할 경우

        return image, tuple(targets)