import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
import torch
from torchvision.ops import batched_nms

def visualize_results(input_image, boxes, labels, scores):
    bccd_labels = ['BG', 'RBC', 'WBC', 'Platelets']

    # NumPy 配列を PyTorch テンソルに変換
    boxes = torch.from_numpy(np.array(boxes)).float()
    labels = torch.from_numpy(np.array(labels)).long()  # 整数型に変更
    scores = torch.from_numpy(np.array(scores)).float()

    if scores is not None:
        boxes = boxes[scores > 0.5]
        labels = labels[scores > 0.5]
        scores = scores[scores > 0.5]

    # 採用する boxes の要素番号が返り値で得られる
    keep = batched_nms(boxes=boxes, scores=scores, idxs=labels, iou_threshold=0.0)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    draw = ImageDraw.Draw(input_image)

    # デフォルトフォントの読み込み
    # font = ImageFont.load_default()

    for box, label in zip(boxes, labels):

        if len(box) == 4:
            box = [box[0], box[1], box[2], box[3]]  # [xmin, ymin, xmax, ymax] の形式
        else:
            continue  # 座標の形式が不正な場合はスキップ

        # box
        draw.rectangle(box, outline='red')
        # label
        label = label.item()
        text = bccd_labels[label]

        # デフォルトフォントを使用してテキストサイズを計算
        # w, h = font.getsize(text)
        draw.rectangle([box[0], box[1], box[0]+16, box[1]+16], fill='red')
        draw.text((box[0], box[1]), text, fill='white')

    return input_image
