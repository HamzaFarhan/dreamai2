from .data import *
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog,\
build_detection_test_loader, DatasetMapper, detection_utils,\
build_detection_train_loader, build_detection_test_loader, build_batch_data_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.structures import BoxMode
from detectron2.modeling import build_model as build_obj_model
import detectron2.utils.comm as comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

def build_obj_opt(cfg, model, lr=0.001):
    cfg.SOLVER.BASE_LR = lr
    return build_optimizer(cfg, model)

def create_dataset_dicts(df, data_dir='', class_names=['person'], channels=3):
    
    dataset_dicts = []
    for i in range(len(df)):
        record = {}
        row = df.iloc[i]
        img_name = row.image
        img_path = str(Path(data_dir)/img_name)
        if channels == 3:
            img = rgb_read(img_path)
        else:    
            img = c1_read(img_path)
        h,w = img.shape[:2]
        record["file_name"] = img_path
        record["image_id"] = i
        record["height"] = h
        record["width"] = w
        labels = row.label.split()
        bboxes = row.bb
        if len(labels) != len(bboxes):
            labels = labels*len(bboxes)
        objs = []
        for b_id,bb in enumerate(bboxes):
            xmin,ymin,xmax,ymax = bb
            obj = {
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": class_names.index(labels[b_id]),
#             "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def obj_dl_mapper(dataset_dict, tfms=instant_tfms(bbox=True, test_tfms=False)):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = rgb_read(str(dataset_dict["file_name"]))
    ann = copy.deepcopy(dataset_dict['annotations'])
    bc = np.array([(a['bbox'], a['category_id']) for a in ann])
    bboxes = bc[:,0].tolist()
    cats = bc[:,1].tolist()
    
    aug = tfms(image=image, bboxes=bboxes, category_ids=cats)
    aug['bboxes'] = list_map(aug['bboxes'], list)
    dataset_dict['image'] = aug['image']

    # if(len(ann) != len(aug['bboxes'])):
    #     print('ooooh')
    #     print(len(ann),len(aug['bboxes']))
    #     temp1 = bb_image(image, bboxes=bboxes, cats=['person']*len(bboxes))
    #     temp2 = bb_image(tensor_to_img(aug['image']), bboxes=aug['bboxes'], cats=['person']*len(aug['bboxes']))
    #     plt_show(temp1)
    #     plt_show(temp2)
    for i in range(len(ann)):
        ann[i]['bbox'] = aug['bboxes'][i]
    dataset_dict['instances'] = detection_utils.annotations_to_instances(ann, image.shape[:2])
    return dataset_dict

class ObjDataLoaders():
    def __init__(self, train=None, valid=None, test=None):
        store_attr(self, 'train,valid,test')

def faster_rcnn_cfg(device=None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = str(device)
    cfg = get_cfg()
    cfg.merge_from_file(
      model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
      )
    )
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
      "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.INPUT.FORMAT = 'RGB'
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.DEVICE = device
    return cfg

def get_obj_dls(df, val_df=None, test_df=None, data_dir='', cfg=faster_rcnn_cfg(),
                tfms=instant_tfms(224, 224, bbox=True), bs=64, shuffle=True,
                class_names=None, num_workers=4, split=True,
                val_size=0.2, test_size=0.15, img_mean=[0.4850, 0.4560, 0.4060],
                img_std=[0.2290, 0.2240, 0.2250], data_name='obj_data', **kwargs):
    
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.PIXEL_MEAN = img_mean
    cfg.MODEL.PIXEL_STD = img_std
    cfg.SOLVER.IMS_PER_BATCH = bs
    
    labels = list(df.iloc[:,2].apply(lambda x: str(x).split()))
    if class_names is None:
        class_names = np.unique(flatten_list(labels))
    class_names = list_map(class_names, str)
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)

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
            
    dset_dicts = [partial(create_dataset_dicts, df=df_, class_names=class_names) for df_ in dfs]
    meta_datas = {}
    for d,dset_d in zip(['train', 'valid', 'test'], dset_dicts):
        dname = f'{data_name}_{d}'
        try:
            DatasetCatalog.register(dname, dset_d)
            MetadataCatalog.get(dname).set(thing_classes=class_names)
            meta_datas[d] = MetadataCatalog.get(dname)
            setattr(cfg.DATASETS, d.upper(), (dname,))
        except:
            DatasetCatalog.clear()
            DatasetCatalog.register(dname, dset_d)
            MetadataCatalog.get(dname).set(thing_classes=class_names)
            meta_datas[d] = MetadataCatalog.get(dname)
            setattr(cfg.DATASETS, d.upper(), (dname,))
        
#     train_metadata = MetadataCatalog.get("obj_data_train")
    mapper = partial(obj_dl_mapper, tfms=transforms_[0])
    dls = [build_detection_train_loader(cfg, mapper)]
    if split:
        mapper = partial(obj_dl_mapper, tfms=transforms_[1])
        dls += [build_detection_test_loader(cfg, f'{data_name}_{d}', mapper) for _,d in zip(dset_dicts[1:],
                                                                                      ['valid', 'test'])]
    dls = ObjDataLoaders(*dls)
    dls.meta_datas = meta_datas
    dls.class_names = class_names
    dls.num_classes = len(class_names)
    dls.suggested_metric = 'AP'
    dls.cfg = cfg
    return dls

def do_obj_eval(dls, model, name='valid', output_folder='obj_outputs'):
    
    model.eval()
    cfg = dls.cfg
    dl = getattr(dls, name)
    dataset_name = getattr(cfg.DATASETS, name.upper())[0]
    cfg.OUTPUT_DIR = output_folder
    output_dir = str(Path(output_folder)/dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    evaluator = COCOEvaluator(dataset_name=dataset_name, cfg=cfg, distributed=False, output_dir=output_dir)
    results = inference_on_dataset(model, dl, evaluator)
    return results

# def bb_image(img, anns, color='green'):
#     if not is_iterable(anns[0]): ans = [anns]
#     for a in anns:
#         a = list_map(a, int)
#         c = color_to_rgb(color)
#         x_min, y_min, x_max, y_max = a
#         cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
#     return img

def bb_df_image(df, index, color='green'):
    row = df.iloc[index]
    img_name = str(row[0])
    anns = row[1]
    cats = row[2].split()
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


def bb_image(image, bboxes, cats, color='red'):
    img = image.copy()
    for bbox, cat in zip(bboxes, cats):
        class_name = cat
        img = visualize_bbox(img, bbox, class_name, color=color)
    return img

def dl_bb_image(dl, index=0, class_names=['person']):

    d = dl.dataset.dataset[index]
    img = rgb_read(d['file_name'])
    ann = d['annotations']
    bbs = [a['bbox'] for a in ann]
    cats = [class_names[a['category_id']] for a in ann]
    img1 = bb_image(img, bboxes=bbs, cats=cats, color='red')
    
    img = tensor_to_img(d['image'])
    f = d['instances'].get_fields()
    bbs = list_map(f['gt_boxes'].tensor, list)
    cats = [class_names[c] for c in f['gt_classes']]
    img2 = bb_image(img, bboxes=bbs, cats=cats, color='blue')
    return img1,img2


