import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec

from box import convert_coordinates

class AnchorBoxes(Layer):
    def __init__(
        self,
        image_height,
        image_width,
        this_scale,
        next_scale,
        aspect_ratios=[0.5, 1.0, 2.0],
        two_boxes_for_ar1=True,
        this_steps=None,
        this_offsets=None,
        clip_boxes=False,
        variances=[0.1, 0.1, 0.2, 0.2],
        coords='centroids',
        **kwargs
    ):
        self.this_scale = this_scale
        self.image_height = image_height
        self.image_width = image_width
        self.next_scale = next_scale
        self.coords = coords
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.aspect_ratios = aspect_ratios
        self.variances = variances
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes

        self.n_boxes = len(aspect_ratios)

        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x): 
        size = min(self.image_height, self.image_width)
        
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        batch_size, feature_map_height, feature_map_width, feature_map_channels = x.shape

        if (self.this_steps is None):
            step_height = self.image_height / feature_map_height
            step_width = self.image_width / feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets
        
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) 
        cy_grid = np.expand_dims(cy_grid, -1) 

        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) 
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) 
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

        variances_tensor = np.zeros_like(boxes_tensor) 
        variances_tensor += self.variances 
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'image_height': self.image_height,
            'image_width': self.image_width,
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DecodeDetections(Layer):
    def __init__(
        self,
        confidence_thresh=0.01,
        iou_threshold=0.45,
        top_k=200,
        nms_max_output_size=400,
        coords='centroids',
        image_height=None,
        image_width=None,
        **kwargs
    ):
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.image_height = image_height
        self.image_width = image_width
        self.coords = coords
        self.nms_max_output_size = nms_max_output_size

        
        self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_image_height = tf.constant(self.image_height, dtype=tf.float32, name='image_height')
        self.tf_image_width = tf.constant(self.image_width, dtype=tf.float32, name='image_width')
        self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')

        super(DecodeDetections, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetections, self).build(input_shape)

    def call(self, y_pred, mask=None):
        cx = y_pred[...,-12] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8] 
        cy = y_pred[...,-11] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7] 
        w = tf.exp(y_pred[...,-10] * y_pred[...,-2]) * y_pred[...,-6] 
        h = tf.exp(y_pred[...,-9] * y_pred[...,-1]) * y_pred[...,-5] 
        
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        def normalized_coords():
            xmin1 = tf.expand_dims(xmin * self.tf_image_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * self.tf_image_height, axis=-1)
            xmax1 = tf.expand_dims(xmax * self.tf_image_width, axis=-1)
            ymax1 = tf.expand_dims(ymax * self.tf_image_height, axis=-1)
            return xmin1, ymin1, xmax1, ymax1
        def non_normalized_coords():
            return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)

        xmin, ymin, xmax, ymax = tf.cond(tf.constant(True), normalized_coords, non_normalized_coords)
        
        y_pred = tf.concat(values=[y_pred[...,:-12], xmin, ymin, xmax, ymax], axis=-1)

        batch_size = tf.shape(y_pred)[0] 
        n_boxes = tf.shape(y_pred)[1]
        n_classes = y_pred.shape[2] - 4
        class_indices = tf.range(1, n_classes)

        def filter_predictions(batch_item):
            def filter_single_class(index):
                confidences = tf.expand_dims(batch_item[..., index], axis=-1)
                class_id = tf.fill(dims=tf.shape(confidences), value=tf.cast(index, dtype=tf.float32))
                box_coordinates = batch_item[...,-4:]

                single_class = tf.concat([class_id, confidences, box_coordinates], axis=-1)
                
                padded_single_class = tf.pad(tensor=single_class,
                                             paddings=[[0, self.tf_nms_max_output_size - tf.shape(len(single_class))[0]], [0, 0]],
                                             mode='CONSTANT',
                                             constant_values=0.0)

                return padded_single_class

            filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
                                                elems=tf.range(1,n_classes),
                                                dtype=tf.float32,
                                                parallel_iterations=128,
                                                back_prop=False,
                                                swap_memory=False,
                                                infer_shape=True,
                                                name='loop_over_classes')

            filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1,6))

            def top_k():
                return tf.gather(params=filtered_predictions,
                                 indices=tf.nn.top_k(filtered_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)
            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=filtered_predictions,
                                            paddings=[[0, self.tf_top_k - tf.shape(filtered_predictions)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[0], self.tf_top_k), top_k, pad_and_top_k)

            return top_k_boxes

        
        output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                  elems=y_pred,
                                  dtype=None,
                                  parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True,
                                  name='loop_over_batch')

        return output_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        return (batch_size, self.tf_top_k, 6) 

    def get_config(self):
        config = {
            'image_height': self.image_height,
            'image_width': self.image_width,
        }
        base_config = super(DecodeDetections, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class L2Normalization(Layer):
    def __init__(self, gamma_init=20, **kwargs):
        if K.image_data_format() == 'channels_last':
            self.axis = 3
        else:
            self.axis = 1
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.gamma_init * np.ones((input_shape[self.axis],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self._trainable_weights = [self.gamma]
        super(L2Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        return output * self.gamma

    def get_config(self):
        config = {
            'gamma_init': self.gamma_init
        }
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
