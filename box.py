import numpy as np

def convert_coordinates(tensor, start_index):
    d = 0

    ind = start_index
    tcopy = np.copy(tensor).astype(np.float)
    tcopy[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 
    tcopy[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 
    tcopy[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] + d 
    tcopy[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] + d 

    return tcopy

def intersection_S(boxes1, boxes2):
    boxes1 = np.expand_dims(boxes1, axis=0)
    boxes2 = np.expand_dims(boxes2, axis=0)

    boxes1 = convert_coordinates(boxes1, start_index=0)
    boxes2 = convert_coordinates(boxes2, start_index=0)

    m = boxes1.shape[0] 
    n = boxes2.shape[0] 

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3
    d = -1

    min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:,[xmin,ymin]], axis=1), reps=(1, n, 1)), np.tile(np.expand_dims(boxes2[:,[xmin,ymin]], axis=0), reps=(m, 1, 1)))
    max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:,[xmax,ymax]], axis=1), reps=(1, n, 1)), np.tile(np.expand_dims(boxes2[:,[xmax,ymax]], axis=0), reps=(m, 1, 1)))
    side_lengths = np.maximum(0, max_xy - min_xy + d)
    
    return side_lengths[:,:,0] * side_lengths[:,:,1]

def iou(boxes1, boxes2):
    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)
    
    boxes1 = convert_coordinates(boxes1, start_index=0)
    boxes2 = convert_coordinates(boxes2, start_index=0)

    intersection_areas = intersection_S(boxes1, boxes2)

    m = boxes1.shape[0] 
    n = boxes2.shape[0] 

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3
    d = 0

    boxes1_areas = np.tile(np.expand_dims((boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d), axis=1), reps=(1,n))
    boxes2_areas = np.tile(np.expand_dims((boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d), axis=0), reps=(m,1))

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas
