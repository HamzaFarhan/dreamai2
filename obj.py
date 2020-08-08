from .data import *

from . import obj_utils as utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset

def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def dai_obj_model(num_classes, pretrained=True, img_mean=None, img_std=None,
                  min_size=256, max_size=256, box_detections_per_img=20, device=None):

    device = default_device(device)
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, image_mean=img_mean, image_std=img_std,
                                    min_size=min_size, max_size=max_size,
                                    box_detections_per_img=box_detections_per_img)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    return model.to(device)

@torch.no_grad()
def evaluate(model, data_batch, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

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
             'iscrowd':torch.zeros((len(row[1]),), dtype=torch.int64)}
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

def collate_obj(batch):
    return tuple(zip(*batch))

def get_obj_dls(df, val_df=None, test_df=None, data_dir='', dset=DaiObjDataset,
                tfms=instant_tfms(256, 256, bbox=True), bs=64, shuffle=True,
                class_names=None, num_workers=4, split=True, pin_memory=True,
                val_size=0.2, test_size=0.15, collate_fn=collate_obj,
                **kwargs):
    
    if tfms is None:
        tfms = albu.Compose([AT.ToTensor()])

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
    dls = DataLoaders(*dls, remove_norm=True)
    # dls = DataLoaders(*dls)
    dls.class_names = class_names
    dls.num_classes = len(class_names)
    dls.suggested_metric = 'AP'
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
