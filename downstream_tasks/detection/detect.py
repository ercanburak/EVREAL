import os
import cv2
import glob
import argparse
import yolov7
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run detection with YOLOv7 model.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image reconstructions directory.')
    parser.add_argument('--output', type=str, required=True, help='Base output folder.')
    parser.add_argument('--img-size', type=int, default=1280, help='Inference size (pixels).')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Object confidence threshold.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS.')
    parser.add_argument('--device', default='cuda', help='Device to run the model on (cuda or cpu).')
    return parser.parse_args()


def process_images(args):
    # Load the YOLOv7 model
    model = yolov7.load('yolov7.pt', device=args.device, trace=False)
    model.conf = args.conf_thres  # NMS confidence threshold
    model.iou = args.iou_thres  # NMS IoU threshold
    model.classes = None  # Optional: filter by class

    # Create output directories
    output_base_path = os.path.join(args.output, os.path.basename(args.input))
    drawings_path = os.path.join(output_base_path, "drawings")
    boxes_path = os.path.join(output_base_path, "boxes")
    os.makedirs(drawings_path, exist_ok=True)
    os.makedirs(boxes_path, exist_ok=True)

    # Load frame list
    with open('frame_list.txt', 'r') as f:
        frame_list = f.read().splitlines()

    # Process each image in the input directory
    input_images = sorted(glob.glob(os.path.join(args.input, "*.png")))
    frame_list = [int(frame) for frame in frame_list]
    input_images = [input_images[i] for i in frame_list]
    min_conf = 1.0
    for img_path in tqdm(input_images):
        frame_id = int(os.path.splitext(os.path.basename(img_path))[0].split('_')[-1])
        img = cv2.imread(img_path)
        results = model(img, size=args.img_size, augment=True)
        detections = results.pred[0].cpu().numpy()  # Detections for the current image

        # Filter detections and draw on images
        for detection in detections:
            x1, y1, x2, y2, conf, cls_pred = map(float, detection[:6])
            if conf < min_conf:
                min_conf = conf
            if cls_pred != 2:  # Assuming class id 2 is for cars
                continue
            label = 'car'
            draw_detection(img, (x1, y1, x2, y2), label, conf)

        # Save output images and detection boxes
        cv2.imwrite(os.path.join(drawings_path, f"frame_{frame_id}.png"), img)
        save_detections(os.path.join(boxes_path, f"frame_{frame_id}.txt"), detections)

    print(f"Minimum confidence: {min_conf:.2f}")


def draw_detection(image, bbox, label, confidence):
    """
    Draw bounding box and label on the image.
    """
    bbox = list(map(int, bbox))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def save_detections(filepath, detections):
    """
    Save detections to a text file.
    """
    with open(filepath, 'w') as f:
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection[:4])
            conf, cls_pred = detection[4:6]
            if int(cls_pred) != 2: continue  # Skip if not a car
            f.write(f"car {conf} {x1} {y1} {x2} {y2}\n")


if __name__ == "__main__":
    args = parse_arguments()
    process_images(args)
