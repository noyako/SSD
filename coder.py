import numpy as np

from box import iou, convert_coordinates


def match_box_order(weight_matrix):
    weight_matrix = np.copy(weight_matrix)
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes))

    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    for _ in range(num_ground_truth_boxes):
        anchor_indices = np.argmax(weight_matrix, axis=1)
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps)
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index

        weight_matrix[ground_truth_index] = 0
        weight_matrix[:,anchor_index] = 0

    return matches


class SSDInputEncoder:
    def __init__(
        self,
        image_height, image_width,
        n_classes,
        predictor_sizes,
        min_scale, max_scale,
        scales,
        aspect_ratios_global,
        aspect_ratios_per_layer,
        two_boxes_for_ar1,
        steps,
        offsets,
        clip_boxes,
        variances,
        matching_type,
        pos_iou_threshold,
        neg_iou_limit,
        border_pixels,
        coords,
        background_id
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes + 1 
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.background_id = background_id
        self.boxes_list = []
        self.wh_list_diag = []
        self.steps_diag = []
        self.offsets_diag = [] 
        self.centers_diag = [] 

        predictor_sizes = np.array(predictor_sizes)
        self.n_boxes = len(aspect_ratios_global)

        
        for i in range(len(self.predictor_sizes)):
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                                                   aspect_ratios=self.aspect_ratios[i],
                                                                                   this_scale=self.scales[i],
                                                                                   next_scale=self.scales[i+1],
                                                                                   this_steps=self.steps[i],
                                                                                   this_offsets=self.offsets[i],
                                                                                   diagnostics=True)
            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)

    def __call__(self, ground_truth_labels, diagnostics=False):
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        batch_size = len(ground_truth_labels)

        y_encoded = self.generate_encoding_template(batch_size=batch_size, diagnostics=False)
        y_encoded[:, :, self.background_id] = 1 
        class_vectors = np.eye(self.n_classes) 

        for i in range(batch_size): 
            if ground_truth_labels[i].size == 0: 
                continue 
            labels = ground_truth_labels[i].astype(np.float) 
            labels[:,[ymin,ymax]] /= self.image_height 
            labels[:,[xmin,xmax]] /= self.image_width 

            labels = convert_coordinates(labels, start_index=xmin)

            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)] 
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin,ymin,xmax,ymax]]], axis=-1) 
            similarities = iou(labels[:,[xmin,ymin,xmax,ymax]], y_encoded[i,:,-12:-8])
            bipartite_matches = match_box_order(weight_matrix=similarities)
            y_encoded[i, bipartite_matches, :-8] = labels_one_hot
            similarities[:, bipartite_matches] = 0

            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        y_encoded[:,:,[-12,-11]] -= y_encoded[:,:,[-8,-7]] 
        y_encoded[:,:,[-12,-11]] /= y_encoded[:,:,[-6,-5]] * y_encoded[:,:,[-4,-3]] 
        y_encoded[:,:,[-10,-9]] /= y_encoded[:,:,[-6,-5]] 
        y_encoded[:,:,[-10,-9]] = np.log(y_encoded[:,:,[-10,-9]]) / y_encoded[:,:,[-2,-1]] 

        return y_encoded

    def generate_anchor_boxes_for_layer(
        self,
        feature_map_size,
        aspect_ratios,
        this_scale,
        next_scale,
        this_steps=None,
        this_offsets=None,
        diagnostics=False
    ):
        size = min(self.image_height, self.image_width)
        
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1):
                
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        if (this_steps is None):
            step_height = self.image_height / feature_map_size[0]
            step_width = self.image_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
        
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) 
        cy_grid = np.expand_dims(cy_grid, -1) 

        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) 
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) 
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] 
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] 

        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0)
        
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.image_width] = self.image_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.image_height] = self.image_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        boxes_tensor[:, :, :, [0, 2]] /= self.image_width
        boxes_tensor[:, :, :, [1, 3]] /= self.image_height

        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0)

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):        
        boxes_batch = []
        for boxes in self.boxes_list:
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        boxes_tensor = np.concatenate(boxes_batch, axis=1)
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances 
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encoding_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template
