import numpy as np
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import os
import sys
from tqdm import tqdm, trange
import h5py
import pickle
from coder import InputEncoder
from box import iou

class RandomCrop:
    def __init__(self, labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.labels_format = labels_format
        self.bound_generator = BoundGenerator(sample_space=((None, None),
                                                            (0.1, None),
                                                            (0.3, None),
                                                            (0.5, None),
                                                            (0.7, None),
                                                            (0.9, None)),
                                              weights=None)
        self.patch_coord_generator = PatchCoordinateGenerator(must_match='h_w',
                                                              min_scale=0.3,
                                                              max_scale=1.0,
                                                              scale_uniformly=False,
                                                              min_aspect_ratio=0.5,
                                                              max_aspect_ratio=2.0)
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion='center_point',
                                    labels_format=self.labels_format)

        self.image_validator = ImageValidator(overlap_criterion='iou',
                                              n_boxes_min=1,
                                              labels_format=self.labels_format,
                                              border_pixels='half')

        self.random_crop = RandomPatchInf(patch_coord_generator=self.patch_coord_generator,
                                          box_filter=self.box_filter,
                                          image_validator=self.image_validator,
                                          bound_generator=self.bound_generator,
                                          n_trials_max=50,
                                          clip_boxes=True,
                                          prob=0.857,
                                          labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.random_crop.labels_format = self.labels_format
        return self.random_crop(image, labels, return_inverter)

class PhotometricDistortions:
    def __init__(self):
        self.convert_RGB_to_HSV = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32 = ConvertDataType(to='float32')
        self.convert_to_uint8 = ConvertDataType(to='uint8')
        self.convert_to_3_channels = ConvertTo3Channels()
        self.random_brightness = RandomBrightness(lower=-32, upper=32, prob=0.5)
        self.random_contrast = RandomContrast(lower=0.5, upper=1.5, prob=0.5)
        self.random_saturation = RandomSaturation(lower=0.5, upper=1.5, prob=0.5)
        self.random_hue = RandomHue(max_delta=18, prob=0.5)
        self.random_channel_swap = RandomChannelSwap(prob=0.01)

        self.sequence = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.random_channel_swap]

    def __call__(self, image, labels):
        for transform in self.sequence:
            image, labels = transform(image, labels)
        return image, labels


class DataAugmentation:
    def __init__(self,
                 image_height=300,
                 image_width=300,
                 background=(123, 117, 104),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        self.labels_format = labels_format

        self.photometric_distortions = PhotometricDistortions()
        self.random_crop = RandomCrop(labels_format=self.labels_format)
        self.random_flip = RandomFlip(dim='horizontal', prob=0.5, labels_format=self.labels_format)

        self.box_filter = BoxFilter(check_overlap=False,
                                    check_min_area=False,
                                    check_degenerate=True,
                                    labels_format=self.labels_format)

        self.sequence = [
            self.photometric_distortions,
            self.expand,
            self.random_crop,
            self.random_flip,
                        ]

    def __call__(self, image, labels, return_inverter=False):
        self.expand.labels_format = self.labels_format
        self.random_crop.labels_format = self.labels_format
        self.random_flip.labels_format = self.labels_format

        inverters = []

        for transform in self.sequence:
            image, labels = transform(image, labels)

        if return_inverter:
            return image, labels, inverters[::-1]
        else:
            return image, labels


class DataGenerator:
    def __init__(self,
                 load_images_into_memory=False,
                 hdf5_dataset_path=None,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        self.labels_output_format = labels_output_format
        self.labels_format = {'class_id': labels_output_format.index('class_id'),
                              'xmin': labels_output_format.index('xmin'),
                              'ymin': labels_output_format.index('ymin'),
                              'xmax': labels_output_format.index('xmax'),
                              'ymax': labels_output_format.index('ymax')}

        self.dataset_size = 0
        self.load_images_into_memory = load_images_into_memory
        self.images = None

        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            if load_images_into_memory:
                self.images = []
                if verbose:
                    it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
                else:
                    it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.filenames = None

        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
        else:
            self.image_ids = None

        if not eval_neutral is None:
            if isinstance(eval_neutral, str):
                with open(eval_neutral, 'rb') as f:
                    self.eval_neutral = pickle.load(f)
            elif isinstance(eval_neutral, (list, tuple)):
                self.eval_neutral = eval_neutral
        else:
            self.eval_neutral = None

        if not hdf5_dataset_path is None:
            self.hdf5_dataset_path = hdf5_dataset_path
            self.load_hdf5_dataset(verbose=verbose)
        else:
            self.hdf5_dataset = None

    def load_hdf5_dataset(self, verbose=True):
        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
        self.dataset_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        if self.load_images_into_memory:
            self.images = []
            if verbose:
                tr = trange(self.dataset_size, desc='Loading images into memory', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading labels', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

        if self.hdf5_dataset.attrs['has_image_ids']:
            self.image_ids = []
            image_ids = self.hdf5_dataset['image_ids']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading image IDs', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.image_ids.append(image_ids[i])

        if self.hdf5_dataset.attrs['has_eval_neutral']:
            self.eval_neutral = []
            eval_neutral = self.hdf5_dataset['eval_neutral']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading evaluation-neutrality annotations', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.eval_neutral.append(eval_neutral[i])

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):

        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if not (self.filenames is None):
                objects_to_shuffle.append(self.filenames)
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids)
            if not (self.eval_neutral is None):
                objects_to_shuffle.append(self.eval_neutral)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        current = 0

        while True:
            batch_X, batch_y = [], []

            if current >= self.dataset_size:
                current = 0
                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if not (self.filenames is None):
                        objects_to_shuffle.append(self.filenames)
                    if not (self.labels is None):
                        objects_to_shuffle.append(self.labels)
                    if not (self.image_ids is None):
                        objects_to_shuffle.append(self.image_ids)
                    if not (self.eval_neutral is None):
                        objects_to_shuffle.append(self.eval_neutral)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]

            batch_indices = self.dataset_indices[current:current + batch_size]
            if not (self.images is None):
                for i in batch_indices:
                    batch_X.append(self.images[i])
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            elif not (self.hdf5_dataset is None):
                for i in batch_indices:
                    batch_X.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.filenames[current:current + batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current + batch_size])
            else:
                batch_y = None

            if not (self.eval_neutral is None):
                batch_eval_neutral = self.eval_neutral[current:current + batch_size]
            else:
                batch_eval_neutral = None

            if not (self.image_ids is None):
                batch_image_ids = self.image_ids[current:current + batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X)
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y)

            current += batch_size
            batch_items_to_remove = []
            batch_inverse_transforms = []

            for i in range(len(batch_X)):
                if not (self.labels is None):
                    batch_y[i] = np.array(batch_y[i])

                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                if transformations:
                    inverse_transforms = []
                    for transform in transformations:
                        batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                        inverse_transforms.append(inverse_transform)
                    batch_inverse_transforms.append(inverse_transforms[::-1])

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):

                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            batch_X = np.array(batch_X)

            if not (label_encoder is None or self.labels is None):

                if ('matched_anchors' in returns) and isinstance(label_encoder, InputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None

            else:
                batch_y_encoded = None
                batch_matched_anchors = None

            ret = []
            if 'processed_images' in returns: ret.append(batch_X)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns: ret.append(batch_original_labels)

            yield ret

    def get_dataset(self):
        return self.filenames, self.labels, self.image_ids, self.eval_neutral

    def get_dataset_size(self):
        return self.dataset_size


class Resize:
    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):
        image_height, image_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin + 1, ymax + 1]] = np.round(
                    labels[:, [ymin + 1, ymax + 1]] * (image_height / self.out_height), decimals=0)
                labels[:, [xmin + 1, xmax + 1]] = np.round(
                    labels[:, [xmin + 1, xmax + 1]] * (image_width / self.out_width), decimals=0)
                return labels

        if labels is None:
            if return_inverter:
                return image, inverter
            else:
                return image
        else:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] * (self.out_height / image_height), decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] * (self.out_width / image_width), decimals=0)

            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.out_height,
                                         image_width=self.out_width)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels


class Flip:
    def __init__(self,
                 dim='horizontal',
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.dim = dim
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):
        image_height, image_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        if self.dim == 'horizontal':
            image = image[:, ::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [xmin, xmax]] = image_width - labels[:, [xmax, xmin]]
                return image, labels
        else:
            image = image[::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [ymin, ymax]] = image_height - labels[:, [ymax, ymin]]
                return image, labels


class RandomFlip:
    def __init__(self,
                 dim='horizontal',
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.dim = dim
        self.prob = prob
        self.labels_format = labels_format
        self.flip = Flip(dim=self.dim, labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0 - self.prob):
            self.flip.labels_format = self.labels_format
            return self.flip(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Scale:
    def __init__(self,
                 factor,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0, 0, 0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.factor = factor
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        image_height, image_width = image.shape[:2]

        M = cv2.getRotationMatrix2D(center=(image_width / 2, image_height / 2),
                                    angle=0,
                                    scale=self.factor)

        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(image_width, image_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            labels = np.copy(labels)

            toplefts = np.array([labels[:, xmin], labels[:, ymin], np.ones(labels.shape[0])])
            bottomrights = np.array([labels[:, xmax], labels[:, ymax], np.ones(labels.shape[0])])
            new_toplefts = (np.dot(M, toplefts)).T
            new_bottomrights = (np.dot(M, bottomrights)).T
            labels[:, [xmin, ymin]] = np.round(new_toplefts, decimals=0).astype(np.int)
            labels[:, [xmax, ymax]] = np.round(new_bottomrights, decimals=0).astype(np.int)

            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=image_height,
                                         image_width=image_width)

            if self.clip_boxes:
                labels[:, [ymin, ymax]] = np.clip(labels[:, [ymin, ymax]], a_min=0, a_max=image_height - 1)
                labels[:, [xmin, xmax]] = np.clip(labels[:, [xmin, xmax]], a_min=0, a_max=image_width - 1)

            return image, labels


class Rotate:
    def __init__(self,
                 angle,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.angle = angle
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        image_height, image_width = image.shape[:2]

        M = cv2.getRotationMatrix2D(center=(image_width / 2, image_height / 2),
                                    angle=self.angle,
                                    scale=1)

        cos_angle = np.abs(M[0, 0])
        sin_angle = np.abs(M[0, 1])

        image_width_new = int(image_height * sin_angle + image_width * cos_angle)
        image_height_new = int(image_height * cos_angle + image_width * sin_angle)

        M[1, 2] += (image_height_new - image_height) / 2
        M[0, 2] += (image_width_new - image_width) / 2

        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(image_width_new, image_height_new))

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            labels = np.copy(labels)

            toplefts = np.array([labels[:, xmin], labels[:, ymin], np.ones(labels.shape[0])])
            bottomrights = np.array([labels[:, xmax], labels[:, ymax], np.ones(labels.shape[0])])
            new_toplefts = (np.dot(M, toplefts)).T
            new_bottomrights = (np.dot(M, bottomrights)).T
            labels[:, [xmin, ymin]] = np.round(new_toplefts, decimals=0).astype(np.int)
            labels[:, [xmax, ymax]] = np.round(new_bottomrights, decimals=0).astype(np.int)

            if self.angle == 90:

                labels[:, [ymax, ymin]] = labels[:, [ymin, ymax]]
            elif self.angle == 180:

                labels[:, [ymax, ymin]] = labels[:, [ymin, ymax]]
                labels[:, [xmax, xmin]] = labels[:, [xmin, xmax]]
            elif self.angle == 270:

                labels[:, [xmax, xmin]] = labels[:, [xmin, xmax]]

            return image, labels


class BoundGenerator:
    def __init__(self,
                 sample_space=((0.1, None),
                               (0.3, None),
                               (0.5, None),
                               (0.7, None),
                               (0.9, None),
                               (None, None)),
                 weights=None):
        self.sample_space = []
        self.sample_space_size = len(self.sample_space)

        if weights is None:
            self.weights = [1.0 / self.sample_space_size] * self.sample_space_size
        else:
            self.weights = weights

    def __call__(self):
        i = np.random.choice(self.sample_space_size, p=self.weights)
        return self.sample_space[i]


class BoxFilter:
    def __init__(self,
                 check_overlap=True,
                 check_min_area=True,
                 check_degenerate=True,
                 overlap_criterion='center_point',
                 overlap_bounds=(0.3, 1.0),
                 min_area=16,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        self.overlap_criterion = overlap_criterion
        self.overlap_bounds = overlap_bounds
        self.min_area = min_area
        self.check_overlap = check_overlap
        self.check_min_area = check_min_area
        self.check_degenerate = check_degenerate
        self.labels_format = labels_format
        self.border_pixels = border_pixels

    def __call__(self,
                 labels,
                 image_height=None,
                 image_width=None):
        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        requirements_met = np.ones(shape=labels.shape[0], dtype=np.bool)

        if self.check_degenerate:
            non_degenerate = (labels[:, xmax] > labels[:, xmin]) * (labels[:, ymax] > labels[:, ymin])
            requirements_met *= non_degenerate

        if self.check_min_area:
            min_area_met = (labels[:, xmax] - labels[:, xmin]) * (labels[:, ymax] - labels[:, ymin]) >= self.min_area
            requirements_met *= min_area_met

        if self.check_overlap:
            if isinstance(self.overlap_bounds, BoundGenerator):
                lower, upper = self.overlap_bounds()
            else:
                lower, upper = self.overlap_bounds

            image_coords = np.array([0, 0, image_width, image_height])
            image_boxes_iou = iou(image_coords, labels[:, [xmin, ymin, xmax, ymax]])
            requirements_met *= (image_boxes_iou > lower) * (image_boxes_iou <= upper)

        return labels[requirements_met]


class ImageValidator:
    def __init__(self,
                 overlap_criterion='center_point',
                 bounds=(0.3, 1.0),
                 n_boxes_min=1,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        self.overlap_criterion = overlap_criterion
        self.bounds = bounds
        self.n_boxes_min = n_boxes_min
        self.labels_format = labels_format
        self.border_pixels = border_pixels
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion=self.overlap_criterion,
                                    overlap_bounds=self.bounds,
                                    labels_format=self.labels_format,
                                    border_pixels=self.border_pixels)

    def __call__(self,
                 labels,
                 image_height,
                 image_width):

        self.box_filter.overlap_bounds = self.bounds
        self.box_filter.labels_format = self.labels_format

        valid_labels = self.box_filter(labels=labels,
                                       image_height=image_height,
                                       image_width=image_width)

        if isinstance(self.n_boxes_min, int):

            if len(valid_labels) >= self.n_boxes_min:
                return True
            else:
                return False
        elif self.n_boxes_min == 'all':

            if len(valid_labels) == len(labels):
                return True
            else:
                return False


class PatchCoordinateGenerator:
    def __init__(self,
                 image_height=None,
                 image_width=None,
                 must_match='h_w',
                 min_scale=0.3,
                 max_scale=1.0,
                 scale_uniformly=False,
                 min_aspect_ratio=0.5,
                 max_aspect_ratio=2.0,
                 patch_ymin=None,
                 patch_xmin=None,
                 patch_height=None,
                 patch_width=None,
                 patch_aspect_ratio=None):
        self.image_height = image_height
        self.image_width = image_width
        self.must_match = must_match
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_uniformly = scale_uniformly
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_aspect_ratio = patch_aspect_ratio

    def __call__(self):
        if self.must_match == 'h_w':
            if not self.scale_uniformly:

                if self.patch_height is None:
                    patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.image_height)
                else:
                    patch_height = self.patch_height

                if self.patch_width is None:
                    patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.image_width)
                else:
                    patch_width = self.patch_width
            else:
                scaling_factor = np.random.uniform(self.min_scale, self.max_scale)
                patch_height = int(scaling_factor * self.image_height)
                patch_width = int(scaling_factor * self.image_width)

        elif self.must_match == 'h_ar':

            if self.patch_height is None:
                patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.image_height)
            else:
                patch_height = self.patch_height

            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio

            patch_width = int(patch_height * patch_aspect_ratio)

        elif self.must_match == 'w_ar':

            if self.patch_width is None:
                patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.image_width)
            else:
                patch_width = self.patch_width

            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio

            patch_height = int(patch_width / patch_aspect_ratio)

        if self.patch_ymin is None:

            y_range = self.image_height - patch_height

            if y_range >= 0:
                patch_ymin = np.random.randint(0, y_range + 1)
            else:
                patch_ymin = np.random.randint(y_range, 1)
        else:
            patch_ymin = self.patch_ymin

        if self.patch_xmin is None:

            x_range = self.image_width - patch_width

            if x_range >= 0:
                patch_xmin = np.random.randint(0, x_range + 1)
            else:
                patch_xmin = np.random.randint(x_range, 1)
        else:
            patch_xmin = self.patch_xmin

        return (patch_ymin, patch_xmin, patch_height, patch_width)


class CropPad:
    def __init__(self,
                 patch_ymin,
                 patch_xmin,
                 patch_height,
                 patch_width,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0, 0, 0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):
        image_height, image_width = image.shape[:2]

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        patch_ymin = self.patch_ymin
        patch_xmin = self.patch_xmin

        if image.ndim == 3:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width, 3), dtype=np.uint8)
            canvas[:, :] = self.background
        elif image.ndim == 2:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width), dtype=np.uint8)
            canvas[:, :] = self.background[0]

        if patch_ymin < 0 and patch_xmin < 0:
            image_crop_height = min(image_height, self.patch_height + patch_ymin)
            image_crop_width = min(image_width, self.patch_width + patch_xmin)
            canvas[-patch_ymin:-patch_ymin + image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[
                                                                                                              :image_crop_height,
                                                                                                              :image_crop_width]

        elif patch_ymin < 0 and patch_xmin >= 0:
            image_crop_height = min(image_height, self.patch_height + patch_ymin)
            image_crop_width = min(self.patch_width, image_width - patch_xmin)
            canvas[-patch_ymin:-patch_ymin + image_crop_height, :image_crop_width] = image[:image_crop_height,
                                                                                     patch_xmin:patch_xmin + image_crop_width]

        elif patch_ymin >= 0 and patch_xmin < 0:
            image_crop_height = min(self.patch_height, image_height - patch_ymin)
            image_crop_width = min(image_width, self.patch_width + patch_xmin)
            canvas[:image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[
                                                                                     patch_ymin:patch_ymin + image_crop_height,
                                                                                     :image_crop_width]

        elif patch_ymin >= 0 and patch_xmin >= 0:
            image_crop_height = min(self.patch_height, image_height - patch_ymin)
            image_crop_width = min(self.patch_width, image_width - patch_xmin)
            canvas[:image_crop_height, :image_crop_width] = image[patch_ymin:patch_ymin + image_crop_height,
                                                            patch_xmin:patch_xmin + image_crop_width]

        image = canvas

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin + 1, ymax + 1]] += patch_ymin
                labels[:, [xmin + 1, xmax + 1]] += patch_xmin
                return labels

        if not (labels is None):

            labels[:, [ymin, ymax]] -= patch_ymin
            labels[:, [xmin, xmax]] -= patch_xmin

            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.patch_height,
                                         image_width=self.patch_width)

            if self.clip_boxes:
                labels[:, [ymin, ymax]] = np.clip(labels[:, [ymin, ymax]], a_min=0, a_max=self.patch_height - 1)
                labels[:, [xmin, xmax]] = np.clip(labels[:, [xmin, xmax]], a_min=0, a_max=self.patch_width - 1)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

        else:
            if return_inverter:
                return image, inverter
            else:
                return image


class Crop:
    def __init__(self,
                 crop_top,
                 crop_bottom,
                 crop_left,
                 crop_right,
                 clip_boxes=True,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.crop = CropPad(patch_ymin=self.crop_top,
                            patch_xmin=self.crop_left,
                            patch_height=None,
                            patch_width=None,
                            clip_boxes=self.clip_boxes,
                            box_filter=self.box_filter,
                            labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        image_height, image_width = image.shape[:2]

        self.crop.patch_height = image_height - self.crop_top - self.crop_bottom
        self.crop.patch_width = image_width - self.crop_left - self.crop_right
        self.crop.labels_format = self.labels_format

        return self.crop(image, labels, return_inverter)


class Pad:
    def __init__(self,
                 pad_top,
                 pad_bottom,
                 pad_left,
                 pad_right,
                 background=(0, 0, 0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.background = background
        self.labels_format = labels_format
        self.pad = CropPad(patch_ymin=-self.pad_top,
                           patch_xmin=-self.pad_left,
                           patch_height=None,
                           patch_width=None,
                           clip_boxes=False,
                           box_filter=None,
                           background=self.background,
                           labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        image_height, image_width = image.shape[:2]

        self.pad.patch_height = image_height + self.pad_top + self.pad_bottom
        self.pad.patch_width = image_width + self.pad_left + self.pad_right
        self.pad.labels_format = self.labels_format

        return self.pad(image, labels, return_inverter)


class RandomPatchInf:
    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 bound_generator=None,
                 n_trials_max=50,
                 clip_boxes=True,
                 prob=0.857,
                 background=(0, 0, 0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.bound_generator = bound_generator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        image_height, image_width = image.shape[:2]
        self.patch_coord_generator.image_height = image_height
        self.patch_coord_generator.image_width = image_width

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        if not self.image_validator is None:
            self.image_validator.labels_format = self.labels_format
        self.sample_patch.labels_format = self.labels_format

        while True:

            p = np.random.uniform(0, 1)
            if p >= (1.0 - self.prob):

                if not ((self.image_validator is None) or (self.bound_generator is None)):
                    self.image_validator.bounds = self.bound_generator()

                for _ in range(max(1, self.n_trials_max)):

                    patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator()

                    self.sample_patch.patch_ymin = patch_ymin
                    self.sample_patch.patch_xmin = patch_xmin
                    self.sample_patch.patch_height = patch_height
                    self.sample_patch.patch_width = patch_width

                    aspect_ratio = patch_width / patch_height
                    if not (
                            self.patch_coord_generator.min_aspect_ratio <= aspect_ratio <= self.patch_coord_generator.max_aspect_ratio):
                        continue

                    if (labels is None) or (self.image_validator is None):

                        return self.sample_patch(image, labels, return_inverter)
                    else:

                        new_labels = np.copy(labels)
                        new_labels[:, [ymin, ymax]] -= patch_ymin
                        new_labels[:, [xmin, xmax]] -= patch_xmin

                        if self.image_validator(labels=new_labels,
                                                image_height=patch_height,
                                                image_width=patch_width):
                            return self.sample_patch(image, labels, return_inverter)
            else:
                if return_inverter:
                    def inverter(labels):
                        return labels

                if labels is None:
                    if return_inverter:
                        return image, inverter
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, inverter
                    else:
                        return image, labels


class ConvertColor:
    def __init__(self, current='RGB', to='HSV', keep_3ch=True):
        self.current = current
        self.to = to
        self.keep_3ch = keep_3ch

    def __call__(self, image, labels=None):
        if self.current == 'RGB' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'RGB' and self.to == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.keep_3ch:
                image = np.stack([image] * 3, axis=-1)
        elif self.current == 'HSV' and self.to == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif self.current == 'HSV' and self.to == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2GRAY)
            if self.keep_3ch:
                image = np.stack([image] * 3, axis=-1)
        if labels is None:
            return image
        else:
            return image, labels


class ConvertDataType:
    def __init__(self, to='uint8'):
        self.to = to

    def __call__(self, image, labels=None):
        if self.to == 'uint8':
            image = np.round(image, decimals=0).astype(np.uint8)
        else:
            image = image.astype(np.float32)
        if labels is None:
            return image
        else:
            return image, labels


class ConvertTo3Channels:
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
        if labels is None:
            return image
        else:
            return image, labels


class Hue:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, image, labels=None):
        image[:, :, 0] = (image[:, :, 0] + self.delta) % 180.0
        if labels is None:
            return image
        else:
            return image, labels


class RandomHue:
    def __init__(self, max_delta=18, prob=0.5):
        self.max_delta = max_delta
        self.prob = prob
        self.change_hue = Hue(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0 - self.prob):
            self.change_hue.delta = np.random.uniform(-self.max_delta, self.max_delta)
            return self.change_hue(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels


class Saturation:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image, labels=None):
        image[:, :, 1] = np.clip(image[:, :, 1] * self.factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels


class RandomSaturation:
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_saturation = Saturation(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0 - self.prob):
            self.change_saturation.factor = np.random.uniform(self.lower, self.upper)
            return self.change_saturation(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels


class Brightness:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, image, labels=None):
        image = np.clip(image + self.delta, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels


class RandomBrightness:
    def __init__(self, lower=-84, upper=84, prob=0.5):
        self.lower = float(lower)
        self.upper = float(upper)
        self.prob = prob
        self.change_brightness = Brightness(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0 - self.prob):
            self.change_brightness.delta = np.random.uniform(self.lower, self.upper)
            return self.change_brightness(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels


class Contrast:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image, labels=None):
        image = np.clip(127.5 + self.factor * (image - 127.5), 0, 255)
        if labels is None:
            return image
        else:
            return image, labels


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_contrast = Contrast(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0 - self.prob):
            self.change_contrast.factor = np.random.uniform(self.lower, self.upper)
            return self.change_contrast(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels


class Gamma:
    def __init__(self, gamma):
        self.gamma = gamma
        self.gamma_inv = 1.0 / gamma

        self.table = np.array([((i / 255.0) ** self.gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def __call__(self, image, labels=None):
        image = cv2.LUT(image, self.table)
        if labels is None:
            return image
        else:
            return image, labels


class HistogramEqualization:
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
        if labels is None:
            return image
        else:
            return image, labels


class ChannelSwap:
    def __init__(self, order):
        self.order = order

    def __call__(self, image, labels=None):
        image = image[:, :, self.order]
        if labels is None:
            return image
        else:
            return image, labels


class RandomChannelSwap:
    def __init__(self, prob=0.5):
        self.prob = prob

        self.permutations = ((0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))
        self.swap_channels = ChannelSwap(order=(0, 1, 2))

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0 - self.prob):
            i = np.random.randint(5)
            self.swap_channels.order = self.permutations[i]
            return self.swap_channels(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels
