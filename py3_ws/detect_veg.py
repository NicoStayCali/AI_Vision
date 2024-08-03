import pyrealsense2 as rs
import numpy as np
import cv2
import supervision as sv
import yolov5
from PIL import Image as Image_pil
import math
# import pcl
# import open3d as o3d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
import torch						#SAM
from segment_anything import build_sam, SamPredictor	#SAM
from segment_anything import sam_model_registry	#SAM
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.

pipe = rs.pipeline()
cfg  = rs.config()

cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
align = rs.align(rs.stream.color)

pipe.start(cfg)

''' # <<SAM>> #
def get_SAM_output(model, image, box):
    model.set_image(image)

    if box.shape[0] > 1:
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
    
def yolo_sam_init():   
    sam_checkpoint = "/home/klaster/AI_Vision/src/veg_pkg/veg_pkg/sam_vit_h_4b8939.pth" 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
'''

model = yolov5.load('/home/klaster/AI_Vision/src/veg_pkg/veg_pkg/yolov5/iceberg.pt')

while True:
    # <<DATA>> #
    frame = pipe.wait_for_frames()
    # depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()
    aligned_frame = align.process(frame)
    aligned_depth_frame = aligned_frame.get_depth_frame()

    # depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.5), cv2.COLORMAP_JET)
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    a_depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha = 0.5), cv2.COLORMAP_JET)
    
    # cv2.imshow('rgb', color_image)
    # cv2.imshow('depth', depth_cm)
    # cv2.imshow('depth', a_depth_cm)
    
    # <<DETECTION>> #
    results = model(color_image)
    detections = sv.Detections.from_yolov5(results)
    
    '''
    result_img = results.render()	#SAM
    img_with_boxes = result_img[0]	#SAM
    bounding_boxes = results.xyxy	#SAM
    '''
    
    for detection in detections:
       x1, y1, x2, y2 = detection[0][:4]
       x1, y1, x2, y2 = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
    
    # <<ANNOTATION>> #
    # polygon_annotator = sv.PolygonAnnotator()
    # poly_annotated_frame = polygon_annotator.annotate(scene=color_image.copy(), detections=detections)
    bounding_box_annotator = sv.BoxAnnotator()
    annotated_frame = bounding_box_annotator.annotate(scene=color_image.copy(), detections=detections)
    det_frame_img = np.zeros((480, 640, 3), np.uint8)
    det_frame_img = Image_pil.fromarray(annotated_frame)
    det_frame_cv2 = np.array(det_frame_img)
    cv2.imshow("image", det_frame_cv2)
    
    if detections:
	    # <<MASK>> #
	    '''
	    yolo_sam_init()								#SAM
	    masks = get_SAM_output(sam_predictor, color_image, bounding_boxes)	#SAM
	    sticker = sticker_generator(masks, image)					#SAM
	    '''
	    
	    mask = np.zeros(color_image.shape[:2], np.uint8)
	    mask[y1:y2, x1:x2] = 255
	    img_masked = cv2.bitwise_and(color_image, color_image, mask=mask)
	    crop_frame_img = np.zeros((480, 640, 3), np.uint8)
	    crop_frame_img = Image_pil.fromarray(img_masked)
	    crop_frame_cv2 = np.array(crop_frame_img)
	    # cv2.imshow("img_crop", crop_frame_cv2)
	    
	    mask_d = np.zeros(aligned_depth_image.shape[:2], np.uint8)
	    mask_d[y1:y2, x1:x2] = 255
	    img_masked_d = cv2.bitwise_and(aligned_depth_image, aligned_depth_image, mask=mask_d)
	    a_depth_cropped = cv2.applyColorMap(cv2.convertScaleAbs(img_masked_d, alpha = 0.5), cv2.COLORMAP_JET)
	    # cv2.imshow("dep_crop", a_depth_cropped)
	    
	    # <<INTRINSICS PARAMS>> #
	    height_im = 480
	    weight_im = 640
	    intrinsics_ppx = 326.5632629394531
	    intrinsics_ppy = 250.20266723632812
	    intrinsics_fx = 606.1911010742188
	    intrinsics_fy = 606.2071533203125
	    depth_scale = 0.001
	    clip_distance_max = 3500
	    
	    # depth = aligned_depth_image * depth_scale
	    depth = img_masked_d * depth_scale
	    rows,cols  = depth.shape
	    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
	    r = r.astype(float)
	    c = c.astype(float)
	    
	    valid = (depth > 0) & (depth < clip_distance_max)
	    valid = np.ravel(valid)
	    z = depth
	    x = z * (c - intrinsics_ppx) / intrinsics_fx
	    y = z * (r - intrinsics_ppy) / intrinsics_fy
	    
	    z = -np.ravel(z)[valid]
	    x = np.ravel(x)[valid]
	    y = np.ravel(y)[valid]
	    pointsxyz = np.vstack((x, y, z)).transpose()
	    
	    # <<POINT CLOUD>> # Serve la nuvola di punti? Non basta interpolare i punti che ho?
	    '''
	    # pcl
	    cloud = pcl.PointCloud()
	    cloud.from_array(pointsxyz.astype(np.float32))
	    visual = pcl.pcl_visualization.CloudViewing()
	    visual.ShowMonochromeCloud(cloud)
	    visual.SpinOnce()
	    '''
	    '''
	    # open3d
	    cloud = o3d.geometry.PointCloud()
	    cloud.points = o3d.utility.Vector3dVector(pointsxyz)
	    
	    cloud.estimate_normals()#(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	    o3d.visualization.draw_geometries([cloud], point_show_normal=True)
	    #print(cloud.normals[0])
	    '''
	    # <<INTERPOLATE>> #

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()
