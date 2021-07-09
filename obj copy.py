from .data import *

# from .pycocotools.coco import COCO
# from .pycocotools.cocoeval import COCOeval

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from collections import defaultdict

class DaiObjDataset(DaiDataset):
    
    def __init__(self, data, data_dir='', tfms=None, channels=3, **kwargs):
        super().__init__(**locals_to_params(locals()))

    def get_x(self, to_tensor=True, **kwargs):
        img = kwargs['img']
        y = kwargs['y']
        bboxes = y['boxes']
        cats = y['labels']
        if self.tfms is not None:
            x = self.tfms(image=img.copy(), bboxes=bboxes, category_ids=cats)
            x['boxes'] = list_map(x['bboxes'], list)
            if self.channels == 1:
                x['image'] = x['image'].unsqueeze(0)
            if not to_tensor:
                x['image'] = tensor_to_img(x['image'])
        else:
            x = {'image':img, 'boxes':bboxes}
        return x
    
    def get_y(self, index, str_to_index=True):
        try:
            row = self.data.iloc[index]
        except:
            row = self.data[index]
        y = {'boxes':row[1], 'labels':row[2], 'image_id':torch.tensor([index]),
             'is_crowd':torch.zeros((len(row[1]),), dtype=torch.int64)}
        if is_str(y['labels']):
            y['labels'] = y['labels'].split()
        show_label = copy.deepcopy(y['labels'])
        if str_to_index:
            if is_str(y['labels'][0]) and hasattr(self, 'class_names'):
                y['labels'] = tensor([self.class_names.index(y_) for y_ in y['labels']])
        return y, show_label
    
    def get_ret(self, **kwargs):
        l = locals_to_params(locals())
        remove_key(l, lambda x: x not in ['x', 'y', 'y2', 'name'])
        l['y']['boxes'] = torch.as_tensor(l['x']['boxes'], dtype=torch.float32)
        boxes = l['y']['boxes']
        l['y']['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])
        l['x'] = l['x']['image']
        l['y']['show_label'] = copy.deepcopy(l['y2'])
        l['y']['name'] = copy.deepcopy(l['name'])
        change_key_name(l, 'y', 'label')
        # change_key_name(l, 'y2', 'show_label')
        return l['x'], l['label']

    def show_data(self, data):
        x,name,label,bb = data[0], data[1]['name'], data[1]['show_label'], data[1]['boxes']
        print(f'Name:{name}')
        if self.tfms is None:
            aug = ''
        else: aug = ' Augmented'
        x = bb_image(x, bb, label)
        plt_show(x, title=f'Normal{aug}: {", ".join(label)}')

def obj_collate(batch):
    data = [b for b in batch]
    return data

def get_obj_dls(df, val_df=None, test_df=None, data_dir='', dset=DaiObjDataset,
                tfms=instant_tfms(256, 256, bbox=True), bs=64, shuffle=True,
                class_names=None, num_workers=4, split=True, pin_memory=True,
                val_size=0.2, test_size=0.15, collate_fn=obj_collate,
                **kwargs):
    
    if tfms is None:
        tfms = albu.Compose([AT.ToTensorV2()])

    if not is_iterable(tfms):
        tfms = [tfms]
    
#     img_mean = None
#     img_std= None
#     for t in tfms:
#         norm, norm_id = get_norm_id(t)
#         if i is not None:
#             del_norm(t, i)
#             img_mean, img_std = norm.mean, norm.std
    
    # labels = list(df.iloc[:,2].apply(lambda x: str(x).split()))
    labels = df.iloc[:,2]
    if is_str(labels[0]):
        labels = list_map(labels, lambda x: str(x).split())
    elif list_or_tuple(labels[0]):
        labels = list_map(labels, lambda x: [str(i) for i in x])
    if class_names is None:
        class_names = np.unique(flatten_list(labels))
    class_names = list_map(class_names, str)
    if 'background' in class_names and 'bg' not in class_names:
        class_names.insert(0, 'bg')
    else:
        class_names.insert(0, 'background')
    stratify_idx = 2
    dfs = [df]
    transforms_ = [tfms[0]]
    if split:
        if val_df is None:
            dfs = list(split_df(df, val_size, stratify_idx=stratify_idx))
            transforms_ = [tfms[0], tfms[1]]
        elif val_df is not None:
            dfs+=[val_df]
            transforms_ = [tfms[0], tfms[1]]
        if (test_size > 0) and (test_df is None):
            val_df, test_df = split_df(dfs[1], test_size, stratify_idx=stratify_idx)
            dfs = [dfs[0], val_df, test_df]
            transforms_ = [tfms[0], tfms[1], tfms[1]]
        elif test_df is not None:
            dfs+=[test_df]
            transforms_ = [tfms[0], tfms[1], tfms[1]]
    else:
        if val_df is not None:
            dfs+=[val_df]
            transforms_ = [tfms[0], tfms[1]]
        if test_df is not None:
            dfs+=[test_df]
            transforms_ = [tfms[0], tfms[1], tfms[1]]
    
    dsets = [dset(data_dir=data_dir, data=df, tfms=tfms_,**kwargs,
                  class_names=class_names,) for df,tfms_ in zip(dfs, transforms_)]
    dls = get_dls(dsets=[dsets[0]], bs=bs, shuffle=shuffle, num_workers=num_workers,
                  pin_memory=pin_memory, collate_fn=collate_fn)
    if split:
        dls += get_dls(dsets=dsets[1:], bs=bs, shuffle=False, num_workers=num_workers,
                       pin_memory=pin_memory, collate_fn=collate_fn)
    # dls = DataLoaders(*dls, remove_norm=True)
    dls = DataLoaders(*dls)
    dls.class_names = class_names
    dls.num_classes = len(class_names)
    dls.suggested_metric = 'MAP'
    return dls

def bb_df_image(df, index, color='green'):
    row = df.iloc[index]
    print(row)
    img_name = str(row[0])
    anns = row[1]
    cats = row[2]
    if is_str(cats):
        cats = cats.split()
    if len(anns) != len(cats):
        cats = cats*len(anns)
    img = rgb_read(img_name)
    img = bb_image(img, anns, color=color, cats=cats)
    return img

def visualize_bbox(img, bbox, class_name='person', color='green', thickness=2):
    
    color = color_to_rgb(color)
    bbox = list_map(bbox, int)
    x_min, y_min, x_max, y_max = bbox
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=(255,255,255), 
        lineType=cv2.LINE_AA,
    )
    return img

def bb_image(image, bboxes, cats, class_names=None, color='red'):
    img = image.copy()
    for bbox, cat in zip(bboxes, cats):            
        class_name = cat
        if class_names is not None:
            class_name = class_names[cat]
        img = visualize_bbox(img, bbox, class_name, color=color)
    return img

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels,
                  true_difficulties=None, class_names=['person'], iou_thresh=0.5, device=None):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    label_map = {k: v + 1 for v, k in enumerate(class_names)}
    # label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    if true_difficulties is None:
        true_difficulties = [tensor([0]*len(x)) for x in true_boxes]

    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
           true_labels) == len(true_difficulties)
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes), dtype=torch.float)  # (n_classes - 1)
    for c in range(n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box.to(device),
                                            object_boxes.to(device))  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of iou_thresh, it's a match
            if max_overlap.item() > iou_thresh:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()
    # mean_average_precision = average_precisions.item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def get_obj_detection(det_batch):
    det_batch = det_batch.copy()
    det_boxes = [o['instances'].get_fields()['pred_boxes'].tensor for o in det_batch]
    det_labels = [o['instances'].get_fields()['pred_classes'] for o in det_batch]
    det_scores = [o['instances'].get_fields()['scores'] for o in det_batch]
    return det_boxes, det_labels, det_scores

def get_obj_true(true_batch):
    true_batch = true_batch.copy()
    true_boxes = [v['instances'].get_fields()['gt_boxes'].tensor for v in true_batch]
    true_labels = [v['instances'].get_fields()['gt_classes'] for v in true_batch]
    return true_boxes, true_labels
