import numpy as np
import cv2
from inference_sdk import InferenceHTTPClient
from inference import get_model
import supervision as sv
from PIL import Image as Image_pil
from segment_anything import build_sam, SamPredictor
import torch
from segment_anything import sam_model_registry
from ultralytics import YOLO
import yolov5

def get_SAM_output(model, image, box):
    model.set_image(image)

    if len(box)>1:#box.shape[0] > 1:
        boxes = torch.Tensor(box, device=model.device)
        transformed_boxes = model.transform.apply_boxes_torch(boxes, image.shape[:2])
        masks, _, _ = model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

    else:
        masks, _, _ = model.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )

    return masks

def get_mask(mask, image):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * image
    mask_image = torch.Tensor(mask_image)
    mask_image_pil = Image_pil.fromarray((mask_image.cpu().numpy()).astype(np.uint8)).convert("RGBA")

    return mask_image_pil

def sticker_generator(masks, image):
    sticker = []
    for mask in masks:
        sticker.append(get_mask(mask, image))

    return sticker

def yolo_sam_detect_veg():

    current_frame = np.zeros((480, 640, 3), np.uint8)
    current_frame = cv2.imread('/home/klaster/AI_Vision/src/veg_pkg/veg_pkg/datasets/Iceberg.v2i.yolov9/valid/images/1_png.rf.b816257b79f4f80982ab1aaab5a1aec3.jpg',cv2.IMREAD_COLOR)

    model = yolov5.load('/home/klaster/AI_Vision/src/veg_pkg/veg_pkg/yolov5/iceberg.pt', device='cpu')
    results = model(current_frame)
    result_img = results.render()
    img_with_boxes = result_img[0]

    bounding_boxes = results.xyxy

    sam_checkpoint = "/home/klaster/AI_Vision/src/veg_pkg/veg_pkg/sam_vit_h_4b8939.pth"
    device = torch.device('cpu')#'cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    DEVICE = torch.device('cpu')#'cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    masks = get_SAM_output(sam_predictor, current_frame, bounding_boxes)
    sticker = sticker_generator(masks, image)
    
    cv2.imshow('rgb', sticker)
    cv2.waitKey(0)


def main(args=None):
    yolo_sam_detect_veg()

if __name__ == '__main__':
    main()
