import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont

def visualize_results(input_image, boxes, labels, scores):
    bccd_labels = ['BG', 'RBC', 'WBC', 'Platelets']

    # `output` の値がリストの場合、NumPy 配列に変換する
    boxes = np.array(boxes)
    labels = np.array(labels)
    scores = np.array(scores) if 'scores' in scores else None

    if scores is not None:
        high_confidence_indices = scores > 0.5
        boxes = boxes[high_confidence_indices]
        labels = labels[high_confidence_indices]

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
        text = bccd_labels[label]

        # デフォルトフォントを使用してテキストサイズを計算
        # w, h = font.getsize(text)
        draw.rectangle([box[0], box[1], box[0]+16, box[1]+16], fill='red')
        draw.text((box[0], box[1]), text, fill='white')

    return input_image
