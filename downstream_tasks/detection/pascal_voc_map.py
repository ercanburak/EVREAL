import os
import numpy as np


def parse_annotation(annotation_line):
    class_str, bbox = annotation_line.split(maxsplit=1)
    xmin, ymin, xmax, ymax = map(float, bbox.split())
    return class_str, [xmin, ymin, xmax, ymax]


def parse_detection(detection_line):
    class_str, remaining = detection_line.split(maxsplit=1)
    confidence, xmin, ymin, xmax, ymax = map(float, remaining.split())
    return class_str, confidence, [xmin, ymin, xmax, ymax]


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def calculate_ap(precision, recall):
    m_precision = [0] * (len(precision) + 2)
    m_recall = [0] * (len(recall) + 2)
    m_recall[-1] = 1

    for i in range(len(precision)):
        m_precision[i+1] = precision[i]
        m_recall[i+1] = recall[i]

    for i in range(len(m_precision) - 2, -1, -1):
        m_precision[i] = max(m_precision[i], m_precision[i+1])

    ap = 0
    for i in range(1, len(m_recall)):
        ap += (m_recall[i] - m_recall[i-1]) * m_precision[i]
    return ap


def voc_ap(gt_boxes, pred_boxes, iou_thresh=0.5):
    # gt_boxes = sorted(gt_boxes, key=lambda x: -x[1])
    pred_boxes = sorted(pred_boxes, key=lambda x: -x[1])
    num_gt = len(gt_boxes)
    assert num_gt > 0

    true_positives = np.zeros(len(pred_boxes))
    false_positives = np.zeros(len(pred_boxes))
    false_negatives = len(gt_boxes)

    for i, pred_box in enumerate(pred_boxes):
        max_iou = -np.inf
        max_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            if gt_box[0] != pred_box[0]:
                continue
            iou = calculate_iou(pred_box[2], gt_box[1])
            if iou > max_iou:
                max_iou = iou
                max_idx = j

        if max_iou >= iou_thresh:
            true_positives[i] = 1
            false_negatives -= 1
            del gt_boxes[max_idx]
        else:
            false_positives[i] = 1

    assert false_negatives >= 0
    cumulative_true_positives = np.cumsum(true_positives)
    cumulative_false_positives = np.cumsum(false_positives)
    recall = cumulative_true_positives / num_gt
    precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
    ap = calculate_ap(precision, recall)
    return ap


def calculate_map(gt_path, pred_path):
    gt_files = os.listdir(gt_path)

    aps = []
    for file in gt_files:
        gt_file = os.path.join(gt_path, file)
        pred_file = os.path.join(pred_path, file)

        with open(gt_file, 'r') as f:
            gt_lines = f.readlines()
            gt_boxes = [parse_annotation(line.strip()) for line in gt_lines]

        with open(pred_file, 'r') as f:
            pred_lines = f.readlines()
            pred_boxes = [parse_detection(line.strip()) for line in pred_lines]

        ap = voc_ap(gt_boxes, pred_boxes)
        aps.append(ap)

    import math
    aps = [0 if math.isnan(x) else x for x in aps]
    mean_ap = np.mean(aps)
    return mean_ap


for model in ['E2VID', 'FireNet', 'E2VID+', 'FireNet+', 'SPADE-E2VID', 'SSL-E2VID', 'ET-Net', 'HyperE2VID', 'groundtruth']:
    gt_path = f'mvsec_nightl21_labels/'
    pred_path = f'outputs/{model}/boxes/'
    map_score = calculate_map(gt_path, pred_path) * 100
    print(f"Mean Average Precision (MAP) for {model}: {map_score:.2f}")
