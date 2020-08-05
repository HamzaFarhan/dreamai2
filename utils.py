from .lars import LARS
from .larc import LARC
from .dai_imports import *
from .randaug import RandAugment

image_extensions = {'.art','.bmp','.cdr','.cdt','.cpt','.cr2','.crw','.djv','.djvu','.erf','.gif','.ico',
                    '.ief','.jng','.jp2','.jpe','.jpeg','.jpf','.jpg','.jpg2','.jpm','.jpx','.nef','.orf',
                    '.pat','.pbm','.pcx','.pgm','.png','.pnm','.ppm','.psd','.ras','.rgb','.svg','.svgz',
                    '.tif','.tiff','.wbmp','.xbm','.xpm','.xwd'}

# DEFAULTS = {'image_extensions': image_extensions}

imagenet_stats = (tensor([0.485, 0.456, 0.406]), tensor([0.229, 0.224, 0.225]))

def default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def np_to_parquet(arr, file_name='test.parquet'):
    arr = list(arr.copy())
    pdf = pd.DataFrame({'data':arr})
    # display(pdf)
    pdf.to_parquet(file_name)
    
def parquet_to_np(file_name='test.parquet'):
    pdf = pd.read_parquet(file_name)
    return np.array(pdf['data'])

def display_img_actual_size(im_data, title=''):
    dpi = 80
    height, width, depth = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.title(title,fontdict={'fontsize':25})
    plt.show()

def plt_show(im, cmap=None, title=''):
    if isinstance(im, torch.Tensor):
        im = tensor_to_img(im)
    plt.imshow(im, cmap=cmap)
    plt.title(title)
    plt.show()

def plt_load(path, show=False, cmap=None, title=''):
    img = plt.imread(path)
    if show:
        plt_show(img, cmap=cmap, title=title)
    return img    

def denorm_img_general(inp, mean=None, std=None):
    inp = inp.numpy()
    inp = inp.transpose((1, 2, 0))
    if mean is None:
        mean = np.mean(inp)
    if std is None:    
        std = np.std(inp)
    inp = std * inp + mean
    inp = np.clip(inp, 0., 1.)
    return inp 

def denorm_img(x, mean=None, std=None):
    if is_tensor(x):
        x = tensor_to_img(x)
    if mean is None:
        mean = np.mean(x)
    if std is None:    
        std = np.std(x)
    mean = mean.numpy()
    std = std.numpy()
    x = std * x + mean
    x =  img_float_to_int(x)
    # x = np.clip(x, 0., 1.)
    return x 

def img_on_bg(img, bg, x_factor=1/2, y_factor=1/2):

    img_h, img_w = img.shape[:2]
    background = Image.fromarray(bg)
    bg_w, bg_h = background.size
    offset = (int((bg_w - img_w) * x_factor), int((bg_h - img_h) * y_factor))
    background.paste(Image.fromarray(img), offset)
    img = np.array(background)
    return img

def bgr2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

def gray2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

def rgb2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def bgra2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)

def img_float_to_int(img):
    return np.clip((np.array(img)*255).astype(np.uint8),0,255)

def img_int_to_float(img):
    return np.clip((np.array(img)/255).astype(np.float),0.,1.)

def rgb_read(img):
    return bgr2rgb(cv2.imread(str(img)))

def c1_read(img):
    return cv2.imread(str(img), 0)

def adjust_lightness(color, amount=1.2):
    color = img_int_to_float(color)
    c = colorsys.rgb_to_hls(*color)
    c = np.array(colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2]))
    return img_float_to_int(c)

def albu_resize(img, h, w, interpolation=1):
    rz = albu.Resize(h, w, interpolation)
    return rz(image=img)['image']

def albu_center_crop(img, h, w):
    cc = albu.CenterCrop(h, w)
    return cc(image=img)['image']

def plot_in_row(imgs,figsize = (20,20),rows = None,columns = None,titles = [],fig_path = 'fig.png',cmap = None):
    fig=plt.figure(figsize=figsize)
    if len(titles) == 0:
        titles = ['image_{}'.format(i) for i in range(len(imgs))]
    if not rows:
        rows = 1
        if columns:
            rows = len(imgs)//columns    
    if not columns:    
        columns = len(imgs)
        if rows:
            columns = len(imgs)//rows
    for i in range(1, columns*rows +1):
        img = imgs[i-1]
        fig.add_subplot(rows, columns, i, title = titles[i-1])
        plt.imshow(img,cmap=cmap)
    fig.savefig(fig_path)    
    plt.show()
    return fig

def get_img_stats(dataset,channels):

    print('Calculating mean and std of the data for standardization. Might take some time, depending on the training data size.')

    imgs = []
    for d in dataset:
        img = d[0]
        imgs.append(img)
    imgs_ = torch.stack(imgs,dim=3)
    imgs_ = imgs_.view(channels,-1)
    imgs_mean = imgs_.mean(dim=1)
    imgs_std = imgs_.std(dim=1)
    del imgs
    del imgs_
    print('Done')
    return imgs_mean,imgs_std

def denorm_tensor(x, img_mean, img_std):
    if x.dim() == 3:
        x.unsqueeze_(0)
    x[:, 0, :, :] = x[:, 0, :, :] * img_std[0] + img_mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * img_std[1] + img_mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * img_std[2] + img_mean[2]
    return x

def next_batch(dl):
    return next(iter(dl))

def filter_modules(net, module):
    return ([(i,n) for i,n in enumerate(list(net.children())) if isinstance(n, module)])

def tensor_to_img(t):
    if t.dim() > 3:
        return [np.array(np.transpose(t_,(1,2,0))) for t_ in t]
    return np.array(np.transpose(t,(1,2,0)))

def smooth_labels(labels,eps=0.1):
    if len(labels.shape) > 1:
        length = len(labels[0])
    else:
        length = len(labels)
    labels = labels * (1 - eps) + (1-labels) * eps / (length - 1)
    return labels

def add_extension_(x, data_path='', ext='.jpg', do_str=True):
    if ext[0] != '.': ext = '.'+ext
    x = Path(data_path)/(x+ext)
    if do_str:
        str(x)
    return x

def add_extension(l, data_path='', ext='.jpg', do_str=True):
    fn = partial(add_extension_, data_path=data_path, ext=ext, do_str=do_str)
    return list_map(l, fn)

def df_more_than_count(df, label='label', count=1):
    return df.copy().groupby(label).filter(lambda x : len(x)>count).reset_index(drop=True)

def df_remove_not_exists(df, col='img'):
    df2 = df.copy()
    for i,img in enumerate(list(df2[col])):
        if not Path(img).exists():
            df2 = df2.drop(i)
    return df2.reset_index(drop=True)

def df_classes(df):
    return np.unique(flatten_list([str(x).split() for x in list(df.iloc[:,1])]))

def split_df(train_df, test_size=0.15, stratify_idx=1, seed=2):
    try:    
        train_df,val_df = train_test_split(train_df, test_size=test_size,
                                           random_state=seed, stratify=train_df.iloc[:,stratify_idx])
    except:
        print('Not stratified.')
        train_df,val_df = train_test_split(train_df, test_size=test_size, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df,val_df  

def df_row_to_cols(df, idx=0):
    return df.copy().rename(columns=df.iloc[0]).drop(df.index[0]).dropna().reset_index(drop=True)

def shift_df_col(df, id1=-1, id2=1):
    cols = df.columns.to_list()
    if id1 < 0:
        id1 = len(cols)+id1
    df2 = df.copy()[[*cols[:id2], cols[id1], *cols[id2:id1], *cols[id1+1:]]]
    return df2

def all_in_list(l1, l2):
    return all(elem in l2  for elem in l1)

def slice_df(df, cols):
    df = df.copy()
    if not is_iterable(cols):
        cols = [cols]
    if is_iterable(cols[0]):
        dfs = []
        for c in cols:
            if not all_in_list(c, df.columns):
                c = [df.columns[i] for i in c]
            dfs.append(df[c])
    else:
        if not all_in_list(cols, df.columns):
            cols = [df.columns[i] for i in cols]
        dfs = [df[cols]]
    df_cols = flatten_list([df_.columns.to_list() for df_ in dfs])
    if df.columns[0] not in df_cols:
        dfs.insert(0, df[df.columns[0]])
    return pd.concat(dfs, axis=1)

def dai_one_hot(labels, class_names):
    return [(np.in1d(list(class_names), l)*1.) for l in labels]

def get_one_hot(df):
    labels = list_map(df.iloc[:,1], lambda x:str(x).split())
    is_multi = np.array(list_map(labels, lambda x:len(x)>1)).any()
    if not is_multi:
        labels = df.iloc[:,1]
    one_hot_labels, class_names = one_hot(labels, multi=is_multi)
    return one_hot_labels, class_names, is_multi

def one_hot(targets, multi=False):
    if multi:
        binerizer = MultiLabelBinarizer()
        dai_1hot = binerizer.fit_transform(targets)
    else:
        binerizer = LabelBinarizer()
        dai_1hot = binerizer.fit_transform(targets)
    return dai_1hot, binerizer.classes_

def folders_to_df(path, imgs_repeat=False):

    imgs = get_image_files(path)
    data = {}
    if not imgs_repeat:
        for img in imgs:
            n = img.name
            p = img.parent.name
            file_path = str(end_of_path(img))
            data[file_path] = p
    else:
        for img in imgs:
            n = img.name
            p = img.parent.name
            data[n] = data.get(n, '') + ' ' + p
        data = {v.split()[0]+'/'+k: v for k, v in data.items()}
    dd = {'imgs': list(data.keys()), 'labels': list(data.values())}
    df = pd.DataFrame(dd, columns=['imgs', 'labels'])
    return df

def locals_to_params(l, omit=[], expand=['kwargs']):
    if 'kwargs' not in expand:
        expand.append('kwargs')
    l = copy.deepcopy(l)
    if 'self' in l.keys():
        del l['self']
    if '__class__' in l.keys():
        del l['__class__']
    keys = dict_keys(l)
    for k in keys:
        if k in expand:
            for k2 in l[k]:
                if k2 not in l.keys():
                    l[k2] = l[k][k2]
            del l[k]
        if k in omit:
            del l[k]
    return l

def merge_dicts(d1,d2):
    d = {}
    for k in d1:
        d[k] = d1[k]
    for k in d2:
        d[k] = d2[k]
    return d

def dict_values(d):
    return list(d.values())

def dict_keys(d):
    return list(d.keys())

def swap_dict_key_letters(d, x, y, strict=False):
    if strict:
        return OrderedDict([(k.replace(x, y), v) if x == k else (k, v) for k, v in d.items()])    
    return OrderedDict([(k.replace(x, y), v) if x in k else (k, v) for k, v in d.items()])

def swap_state_dict_key_first_letter(sd, x, y):
    return OrderedDict([(y+k[1:], v) if k[0]==x else (k, v) for k, v in sd.items()])

def change_key_name(d, k1, k2):
    d[k2] = d[k1]
    del d[k1]

def remove_key(d, fn):
    if is_str(fn):
        t = copy.deepcopy(fn)
        fn = lambda x: x==t
    keys = dict_keys(d)
    for k in keys:
        if fn(k):
            del d[k]

def checkpoint_to_model(checkpoint, only_body=False, only_head=False, swap_x='', swap_y=''):
    model_sd = swap_dict_key_letters(checkpoint['model'], swap_x, swap_y)
    if only_body:
        remove_key(model_sd, lambda x: x.startswith('1.'))
    elif only_head:
        remove_key(model_sd, lambda x: x.startswith('0.'))
    return model_sd

def split_params(model, n=3):
    return list_map(np.array_split(params(model), n), list)

def is_bce(x):
    return isinstance(x, nn.BCEWithLogitsLoss)

def is_cross_entropy(x):
    return isinstance(x, nn.CrossEntropyLoss)

def is_lars(x):
    return isinstance(x, LARS)

def is_larc(x):
    return isinstance(x, LARC)

def model_is_cuda(model):
    return next(model.parameters()).is_cuda

def is_list(x):
    return isinstance(x, list)

def is_tuple(x):
    return isinstance(x, tuple)

def list_or_tuple(x):
    return (is_list(x) or is_tuple(x))

def is_iterable(x):
    return list_or_tuple(x) or is_array(x)

def is_dict(x):
    return isinstance(x, dict)

def is_df(x):
    return isinstance(x, pd.core.frame.DataFrame)

def is_str(x):
    return isinstance(x, str)

def is_int(x):
    return isinstance(x, int)    

def is_array(x):
    return isinstance(x, np.ndarray)

def is_pilimage(x):
    return 'PIL' in str(type(x))

def is_tensor(x):
    return isinstance(x, torch.Tensor)

def is_set(x):
    return isinstance(x, set)

def is_path(x):
    return isinstance(x, Path)

def path_or_str(x):
    return is_str(x) or is_path(x)

def is_norm(x):
    return type(x).__name__ == 'Normalize'

def get_norm(tfms):
    try:
        tfms_list = list(tfms)
    except:
        tfms_list = list(tfms.transforms)
    for t in tfms_list:
        if is_norm(t):
            return t
    return False

def get_norm_id(tfms):
    try:
        tfms_list = list(tfms)
    except:
        tfms_list = list(tfms.transforms)
    for i,t in enumerate(tfms_list):
        if is_norm(t):
            return t,i
    return None, None

def del_norm(tfms, idx=None):
    if idx is None:
        idx = get_norm_id(tfms)[1]
    if idx is not None:
        del tfms.transforms.transforms[idx]

def is_device(x):
    return isinstance(x, torch.device)

def is_module_list(x):
    return isinstance(x, nn.ModuleList)

def is_subscriptable(x):
    return hasattr(x, '__getitem__')

def is_sequential(x):
    return isinstance(x, nn.Sequential)

def is_resize(x):
    return ('Resize' in type(x).__name__) or ('resize' in type(x).__name__)

def get_resize(tfms):
    try:
        tfms_list = list(tfms)
    except:
        tfms_list = list(tfms.transforms)
    for t in tfms_list:
        if is_resize(t):
            return t
    return False

def is_shuffle(dl):
    return isinstance(dl.sampler, RandomSampler)

def new_dl(dl_func, bs=None):
    if bs is not None:
        dl_func = partial(dl_func, batch_size=bs)
    return dl_func()

def set_resize_dims(tfms, h=224, w=224):
    list(tfms)[0].height = h
    list(tfms)[0].width = w

def has_norm(tfms):
    tfms_list = list(tfms)
    for t in tfms_list:
        if is_norm(t):
            return True
    return False

def to_tensor(x):
    t = AT.ToTensor()
    if type(x) == list:
        return [t(image=i)['image'] for i in x]
    return t(image=x)['image']

def batchify_dict(d):
    for k in d.keys():
        d[k].unsqueeze_(0)

def store_attr(self, nms):
    "Store params named in comma-separated `nms` from calling context into attrs in `self`"
    mod = inspect.currentframe().f_back.f_locals
    for n in re.split(', *', nms): setattr(self,n,mod[n])

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

def load_state_dict(model, sd, strict=True, eval=True):
    model.load_state_dict(sd, strict=strict)
    if eval:
        model.eval()

def set_lr(opt, lr, idx=None):
    if idx is None:
        l = len(opt.param_groups) 
        if l <= 10:
            idx = [-1]
        else:
            idx = range(l)
    elif not list_or_tuple(idx):
        idx = [idx]
    for i in idx:
        opt.param_groups[i]['lr'] = lr

def get_lr(opt):
    # return opt.param_groups[0]['lr']
    return opt.param_groups[-1]['lr']

def num_batches(dl):
    try:
        n = len(dl)
    except:
        n = int(len(dl.dataset.dataset)/dl.batch_size)+1
    return n

def get_optim(optimizer_name,params,lr):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(params=params,lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(params=params,lr=lr)
    elif optimizer_name.lower() == 'adadelta':
        return optim.Adadelta(params=params)

def freeze_params(params):
    for p in params:
        p.requires_grad = False

def unfreeze_params(params):
    for p in params:
        p.requires_grad = True

def freeze_model(model):
    freeze_params(model.parameters())

def unfreeze_model(model):
    unfreeze_params(model.parameters())

def params(m):
    "Return all parameters of `m`"
    return [p for p in m.parameters()]

def instant_tfms(h=224, w=224, resize=albu.Resize, test_resize=albu.Resize, bbox=False,
                 tensorfy=True, img_mean=None, img_std=None, extra=[], test_tfms=True):
    normalize,t  = None, None
    if img_mean is not None:
        normalize = albu.Normalize(img_mean, img_std)
    if tensorfy:
        t = AT.ToTensor()

    tfms1 = [[resize(height=h, width=w), *extra, normalize, t]]
    tfms2 = [[test_resize(height=h, width=w), normalize, t]]
    if bbox:
        tfms1.append(albu.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0))
        tfms2.append(albu.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0))
    tfms1 = albu.Compose(*tfms1)
    tfms2 = albu.Compose(*tfms2)
    if test_tfms:
        return tfms1, tfms2
    return tfms1

def dai_tfms(h=224, w=224, resize=albu.Resize, test_resize=albu.Resize, bbox=False,
             tensorfy=True, img_mean=None, img_std=None, extra=[], color=True, test_tfms=True):

    color_tfms = [albu.HueSaturationValue(p=0.3)]
    distortion = [albu.OneOf([albu.OpticalDistortion(p=0.3),
                              albu.GridDistortion(p=.1),
                              albu.IAAPiecewiseAffine(p=0.3),],
                              p=0.2)]
    extra += [
        albu.RandomRotate90(),
        albu.Flip(),
        albu.Transpose(),
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),
            albu.GaussNoise(),
            albu.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True)
        ], p=0.2),
        albu.OneOf([
            albu.MotionBlur(p=.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(),            
        ], p=0.3)
    ]
    if not bbox:
        extra += distortion
    if color:
        extra += color_tfms
    normalize,t  = None, None
    if img_mean is not None:
        normalize = albu.Normalize(img_mean, img_std)
    if tensorfy:
        t = AT.ToTensor()
    tfms1 = [[resize(height=h, width=w), *extra, normalize, t]]
    tfms2 = [[test_resize(height=h, width=w), normalize, t]]
    if bbox:
        tfms1.append(albu.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0))
        tfms2.append(albu.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0))
    tfms1 = albu.Compose(*tfms1)
    tfms2 = albu.Compose(*tfms2)
    if test_tfms:
        return tfms1, tfms2
    return tfms1

def jigsaw_tfms(tfms1, tfms2=None, index=-2):
    tfms = copy.deepcopy(tfms1)
    # tfms.transforms.transforms.insert(0, albu.RandomGridShuffle(p=1.))
    tfms.transforms.transforms.insert(index, albu.RandomGridShuffle(p=1.))
    if tfms2 is not None:
        return tfms, tfms2
    return tfms

def rand_aug(h=224,w=224, resize=transforms.Resize, test_resize=transforms.Resize,
             tensorfy=True, img_mean=None, img_std=None, aug_n=4, aug_m=1):
    
    extra = RandAugment(aug_n, aug_m)
    normalize,t  = noop, noop
    if img_mean is not None:
        normalize = transforms.Normalize(img_mean, img_std)
    if tensorfy:
        t = transforms.ToTensor()
    tfms1 = [resize((h,w)), t, normalize]
    tfms2 = [test_resize((h,w)), t, normalize]
    tfms1 = transforms.Compose(tfms1)
    tfms1.transforms.insert(1, extra)
    tfms2 = transforms.Compose(tfms2)
    return tfms1, tfms2

# def dai_tfms(h=224,w=224, tensorfy=True, img_mean=None, img_std=None):
#     # t1 = [albu.HorizontalFlip(), albu.Rotate(10.), albu.ShiftScaleRotate(0,0.15,0),
#     #       albu.RandomBrightnessContrast(0.1, 0.1), albu.ShiftScaleRotate(0.03,0,0)]
#     t1 = [albu.HorizontalFlip(), albu.Rotate(10.), albu.ShiftScaleRotate(0,0.15,0), albu.ShiftScaleRotate(0.03,0,0)]
#     t2 = list(instant_tfms(h, w, tensorfy, img_mean, img_std))
#     return albu.Compose(t1+t2)

def apply_tfms(img, tfms):
    try:
        if is_array(img):
            img = Image.fromarray(img)
        return tfms(img)
    except:
        if is_pilimage(img):
            img = np.array(img)
        return tfms(image=img)['image']

def imgs_to_batch(paths = [], imgs = [], bs = 1, size = None, norm = False, img_mean = None, img_std = None,
                  stats_percentage = 1., channels = 3, num_workers = 6):
    if len(paths) > 0:
        data = pd.DataFrame({'Images':paths})
    elif len(imgs) > 0:
        data = pd.DataFrame({'Images':imgs})
    tfms = [AT.ToTensor()]
    if norm:
        if img_mean is None:
            norm_tfms = albu.Compose(tfms)
            frac_data = data.sample(frac = stats_percentage).reset_index(drop=True).copy()
            temp_dataset = imgs_to_batch_dataset(data = frac_data, transforms_ = norm_tfms, channels = channels)
            img_mean,img_std = get_img_stats(temp_dataset,channels)
        tfms.insert(0,albu.Normalize(img_mean,img_std))
    if size:
        tfms.insert(0,albu.Resize(size[0],size[1],interpolation=0))        
    tfms = albu.Compose(tfms)
    image_dataset = imgs_to_batch_dataset(data = data, transforms_ = tfms, channels = channels)
    if size is None:
        loader = None
    else:
        loader = DataLoader(image_dataset, batch_size = bs, shuffle=True, num_workers = num_workers)
    return image_dataset,loader

# def imgs_to_batch_old(paths = [],imgs = [], size = None, smaller_factor = None, enlarge_factor = None, mean = None, std = None,
#                   stats_percentage = 1.,show = False, norm = False, bgr_to_rgb = False, device = None, channels = 3):
#     tfms = [AT.ToTensor()]    
#     if len(paths) > 0:
#         if channels == 3:
#             bgr_to_rgb = True
#             imgs = []
#             for p in paths:
#                 imgs.append(cv2.imread(str(p)))
#         elif channels == 1:
#             imgs = []
#             for p in paths:
#                 imgs.append(cv2.imread(str(p),0))
#     if size:
#         tfms.insert(0,albu.Resize(size[0],size[1],interpolation=0))        
#     if norm:
#         norm_tfms = albu.Compose(tfms)
#         if mean is None:
#             mean,std = get_img_stats(imgs,norm_tfms,channels,stats_percentage)
#         tfms.insert(1,albu.Normalize(mean,std))
#     for i,img in enumerate(imgs):
#         if bgr_to_rgb:
#             img = bgr2rgb(img)
#         if show:
#             cmap = None
#             if channels == 1:
#                 cmap = 'gray'
#             try:
#                 plt_show(img,cmap=cmap)   
#             except:
#                 plt_show(img,cmap=cmap)
        
#         transform = albu.Compose(tfms)
#         x = transform(image=img)['image']
#         if channels == 1:
#             x.unsqueeze_(0)
#         imgs[i] = x
#     batch = torch.stack(imgs, dim=0)
#     if device is not None:
#         batch = batch.to(device)
#     return batch

def to_batch(paths = [],imgs = [], size = None, channels=3):
    if len(paths) > 0:
        imgs = []
        for p in paths:
            if channels==3:
                img = bgr2rgb(cv2.imread(p))
            elif channels==1:
                img = cv2.imread(p,0)
            imgs.append(img)
    for i,img in enumerate(imgs):
        if size:
            img = cv2.resize(img, size)
        img =  img.transpose((2,0,1))
        imgs[i] = img
    return torch.from_numpy(np.asarray(imgs)).float()    

def batch_to_imgs(batch,mean=None,std=None):
    imgs = []
    for i in batch:
        if mean is not None:
            imgs.append(denorm_img_general(i,mean,std))
        else:
            imgs.append(i)
    return imgs    

def mini_batch(dataset,bs,start=0):
    imgs = torch.Tensor(bs,*dataset[0][0].shape)
    s = dataset[0][1].shape
    if len(s) > 0:
        labels = torch.Tensor(bs,*s)
    else:    
        labels = torch.Tensor(bs).int()
    for i in range(start,bs+start):
        b = dataset[i]
        imgs[i-start] = b[0]
        labels[i-start] = tensor(b[1])
    return imgs,labels

def pad_list(x, n):
    y = x
    if is_list(y[0]):
        y+=list_map(list(np.tile(y[-1], (n,1))), list)
    else:
        y+=list(np.tile(y[-1], (n)))
    # elif isinstance(y[0], Path):
        # y+=list_map(list(np.tile(y[-1], (n,1))), Path)
    return y

def items_to_idx(l1, l2):
    ret = []
    for i in l1:
        ret.append(l2.index(i))
    return ret

DAI_AvgPool = nn.AdaptiveAvgPool2d(1)

def flatten_tensor(x):
    return x.view(x.shape[0],-1)

def flatten_list(l):
    try:
        return sum(l, [])
    except:
        return sum(l, ())

def rmse(inputs,targets):
    return torch.sqrt(torch.mean((inputs - targets) ** 2))

def psnr(mse):
    return 10 * math.log10(1 / mse)

def get_psnr(inputs,targets):
    mse_loss = F.mse_loss(inputs,targets)
    return 10 * math.log10(1 / mse_loss)

def bce_loss_func(out, targ):
    return nn.BCEWithLogitsLoss()(out, targ.float())

def remove_bn(s):
    for m in s.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()

def is_pool_type(l): return re.search(r'Pool[123]d$', l.__class__.__name__)

def has_pool_type(m):
    "Return `True` if `m` is a pooling layer or has one in its children"
    if is_pool_type(m): return True
    for l in m.children():
        if has_pool_type(l): return True
    return False

def dice(input, targs, iou = False, eps = 1e-8):
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input * targs).sum(dim=1).float()
    union = (input+targs).sum(dim=1).float()
    if not iou: l = 2. * intersect / union
    else: l = intersect / (union-intersect+eps)
    l[union == 0.] = 1.
    return l.mean()

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = F.sigmoid(pred.argmax(dim=1).view(num, -1).float())  # Flatten
    m2 = F.sigmoid(target.view(num, -1).float())  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def get_frames(video_path, start = None, stop = None):

    if start is None:
        start = 0
    vs = cv2.VideoCapture(str(video_path))
    vs.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_number = start
    frames = []
    while True:  
        if stop is not None:
            if frame_number >= stop:
                break
        (grabbed, frame) = vs.read()
        if grabbed:
            frame_number+=1
        if not grabbed:
            break
        frame = bgr2rgb(frame)
        frames.append(frame.astype(float)/255.)
    frames = frames[start:]
    vs.release()
    return frames

def save_imgs(imgs, dest_path = '', img_name = ''):

    if len(img_name) == 0:
        img_name = 'img'
    dest_path = Path(dest_path)
    os.makedirs(dest_path, exist_ok=True)
    for i,img in enumerate(imgs):
        plt.imsave(str(dest_path/f'{img_name}_{i}.png'), img)

def frames_to_vid(frames=[], frames_folder='', output_path='', fps=30):

    os.makedirs(Path(output_path).absolute().parent, exist_ok=True)
    if len(frames) == 0:
        frames_path = path_list(Path(frames_folder))
        first_frame = bgr2rgb(cv2.imread(str(frames_path[0])))
        height, width, _ = first_frame[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for frame_path in frames_path:
            frame = bgr2rgb(cv2.imread(str(frame_path)))
            out.write(bgr2rgb(np.uint8(frame*255)))
        out.release()
    else:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames:
            out.write(bgr2rgb(np.uint8(frame*255)))
        out.release()

def split_video(video_path, start=0, stop=None, output_path='split_video.mp4', fps=30):

    os.makedirs(Path(output_path).absolute().parent, exist_ok=True)
    if start is None:
        start = 0
    vs = cv2.VideoCapture(str(video_path))
    vs.set(cv2.CAP_PROP_POS_FRAMES, start)
    (grabbed, frame) = vs.read()
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_number = start
    while True:  
        if stop is not None:
            if frame_number >= stop:
                break
        (grabbed, frame) = vs.read()
        if grabbed:
            frame_number+=1
        if not grabbed:
            break
        frame = bgr2rgb(frame).astype(float)/255.
        out.write(bgr2rgb(np.uint8(frame*255)))
    out.release()
    vs.release()

def vid_to_frames(v, dest_folder='', name='frame%05d.jpg', fps=30):
    os.makedirs(dest_folder, exist_ok=True)
    vp = Path(dest_folder)
    if isinstance(v, str) or isinstance(v, Path):
        vid = editor.VideoFileClip(str(v))
    else:
        vid = v
    imgs = vid.write_images_sequence(str(vp/name), fps=fps)
    return imgs

def extract_frames(v, t0=0, t1=None, fps=30):
    if path_or_str(v):
        vid = editor.VideoFileClip(str(v))
    else:
        vid = v
    return list(vid.subclip(t0,t1).iter_frames(fps))

def vid_folders_to_frames(video_dict, video_path='videos', frame_path='frames',
                          frame_name='frame%05d.jpg', fps=30):
    
    folders, vids = list(video_dict.keys()), list(video_dict.values())
    for i,f in enumerate(folders):
        fp = Path(frame_path)/f
        os.makedirs(fp, exist_ok=True)
        for v in vids[i]:
            vp = fp/v[:-4]
            os.makedirs(vp, exist_ok=True)
            imgs = vid_to_frames(str(video_path/f/v), dest_folder=vp, name=frame_name, fps=fps)
    #         vid = editor.VideoFileClip(str(video_path/f/v))
    #         imgs = vid.write_images_sequence(str(vp/'frame%05d.jpg'), fps=10)

def add_text(img, text, x_factor=2, y_factor=2, font=cv2.FONT_HERSHEY_SIMPLEX, scale=5.,
                color='white', thickness=10):
    color = color_to_rgb(color)
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]
    textX = ((img.shape[1] - textsize[0]) // x_factor)
    textY = img.shape[0] - (((img.shape[0] - textsize[1]) // y_factor))
    img = cv2.putText(img, text, (textX, textY), font, scale, color, thickness)
    return img

def add_text_pil(img, text=['DreamAI'], x=None, y=None, font='verdana', font_size=None,
                 color='white', stroke_width=0, stroke_fill='blue', align='center', bg=None):

    if is_str(text):
        text = [text]
    x_,y_ = x,y
    offset = 0
    if is_str(img):
        img = Image.open(img)
    elif isinstance(img, Path):
        img = Image.open(str(img))
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    d = ImageDraw.Draw(img)
    sizes = []
    for i,txt in enumerate(text):
        if font_size is None:
            font_size = img.size[1]
            s = img.size
            while not ((np.array(s)*5) < img.size).all():
                font_size -= int(img.size[1]/10)
                fnt = ImageFont.truetype(get_font(font), font_size)
                # d = ImageDraw.Draw(img)
                s = fnt.getsize(txt)
        else:
            fnt = ImageFont.truetype(get_font(font), font_size)
            # d = ImageDraw.Draw(img)
            s = fnt.getsize(txt)
        sizes.append(s)
    sizes = np.array(sizes)
    if bg is not None:
        bg_w = sizes[:,0].max()+20
        bg_h = sizes[:,1].sum()*2
        if x_ is None:
            x = (img.size[0]//2) - (bg_w//2)
        if y_ is None:
            y = (img.size[1]//2) - (bg_h//2)
        d.rectangle(((x,y),(x+bg_w, y+bg_h)), fill=bg)
    
    for i,data in enumerate(zip(text, sizes)):
        txt,s = data
        if i > 0:
            offset += int(s[1]*1.5)
        if x_ is None:
            x = (img.size[0]//2) - (s[0]//2)
        if y_ is None:
            y = (img.size[1]//2) - (s[1]//2) - (s[1]*(len(text)-1))
        d.text((x, y+offset), txt, font=fnt, fill=color, align=align, stroke_width=stroke_width, stroke_fill=stroke_fill)
    img = np.array(img)
    return img

def text_size(txt, font='dejavu serif', font_size=10):

    fnt = ImageFont.truetype(get_font(font), font_size)
    s = np.array((0,0))
    if not list_or_tuple(txt):
        txt = [txt]
    for t in txt:
        s += fnt.getsize(t)
    return s

def remove_from_list(l, r):
    for x in r:
        if x in l:
            l.remove(x)
    return l

def num_common(l1, l2):
    return len(list(set(l1).intersection(set(l2))))

def max_n(l, n=3):
    a = np.array(l)
    idx = heapq.nlargest(n, range(len(a)), a.take)
    return idx, a[idx]

def k_dominant_colors(img, k):

    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters = k)
    clt.fit(img)
    return clt.cluster_centers_

def solid_color_img(shape=(300,300,3), color='black'):
    image = np.zeros(shape, np.uint8)
    color = color_to_rgb(color)
    image[:] = color
    return image

def color_to_rgb(color):
    if type(color) == str:
        return list_map(np.ceil(colors.to_rgb(color)).astype(int)*255, int)
    return list_map(color, int)

def get_font(font):
    fonts = [f.fname for f in matplotlib.font_manager.fontManager.ttflist if ((font.lower() in f.name.lower()) and not ('italic' in f.name.lower()))]
    if len(fonts) == 0:
        fonts = [f.fname for f in matplotlib.font_manager.fontManager.ttflist if (('serif' in f.name.lower()) and not ('italic' in f.name.lower()))]
        print(f'"{font.capitalize()}" font not found. Using "{Path(fonts[0]).stem.capitalize()}"')

    return fonts[0]

def expand_rect(left,top,right,bottom,H,W, margin = 15):
    if top >= margin:
        top -= margin
    if left >= margin:
        left -= margin
    if bottom <= H-margin:
        bottom += margin
    if right <= W-margin:
        right += margin
    return left,top,right,bottom

def show_landmarks(image, landmarks):
    landmarks = np.array(landmarks)    
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def chunkify(l, chunk_size):

    if list_or_tuple(chunk_size):
        l2 = []
        l2.append(l[:chunk_size[0]])
        for i in range(1, len(chunk_size)):
            c1 = sum(chunk_size[:i])
            c2 = chunk_size[i]+c1
            l2.append(l[c1:c2])
        return l2

    return [l[i:i+chunk_size] for i in range(0,len(l), chunk_size)]

def setify(o): return o if isinstance(o,set) else set(list(o))

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    if folders is None:
        folders = list([])
    path = Path(path)
    if extensions is not None:
        extensions = setify(extensions)
        extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(folders) !=0 and i==0: d[:] = [o for o in d if o in folders]
            else:                         d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) !=0 and i==0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return list(res)

def get_image_files(path, recurse=True, folders=None, map_fn=None, sort_key=None):
    "Get image files in `path` recursively, only in `folders`, if specified."
    l = get_files(path, extensions=image_extensions, recurse=recurse, folders=folders)
    if sort_key is not None:
        l = sorted(l, key=sort_key)
    if map_fn is not None:
        return list_map(l, map_fn)
    return l

def path_name(x):
    return x.name

def last_modified(x):
    return x.stat().st_ctime

def list_map(l, m):
    return list(pd.Series(l).apply(m))
    # return [m(x) for x in l]

def p_list(path):
    return list(Path(path).iterdir())

def path_list(path, suffix=None, make_str=False, map_fn=noop):
    # if sort:
    #     if suffix is None:
    #         l = sorted(list(Path(path).iterdir()))
    #         # return sorted(list(Path(path).iterdir()))
    #     else:
    #         l = sorted([p for p in list(Path(path).iterdir()) if p.suffix==suffix])
    #     # return sorted([p for p in list(Path(path).iterdir()) if p.suffix==suffix])
    # else:
    if suffix is None:
        l = p_list(path)
        # return list(Path(path).iterdir())
    else:
        l = [p for p in list(Path(path).iterdir()) if p.suffix==suffix]
    l = list_map(l, map_fn) 
    if make_str:
        l = list_map(l, str)
    return l

def sorted_paths(path, key=None, suffix=None, make_str=False, map_fn=None,
                 reverse=False, only_dirs=False, full_path=True):

    if suffix is None:
        l = p_list(path)
    else:
        if isinstance(suffix, str):
            suffix = (suffix)
        l = [p for p in p_list(path) if p.suffix in suffix]
    if only_dirs:
        l = [x for x in l if x.is_dir()]
    if key is None:
        l = sorted(l, key=last_modified, reverse=True)
    else:
        l = sorted(l, key=key, reverse=reverse)
    if map_fn is not None:
        l = list_map(l, map_fn)
    if not full_path:
        l = list_map(l, lambda x:x.name)
    if make_str:
        l = list_map(l, str)
    return l

def folders_with_files(p, full_path=False, folder_sort_key=None, file_sort_key=None, suffix=None, num_files=None,
                       folder_key=lambda x:x, make_str=False, map_fn=noop, reverse=False):
    
    folders = sorted_paths(p, key=folder_sort_key, reverse=reverse, only_dirs=True)
    folders_dict = dict()
    for f in folders:
        # if f.is_dir():
        if full_path:
            folders_dict[folder_key(f.name)] = sorted_paths(f, key=file_sort_key, suffix=suffix,
                                                            make_str=make_str, reverse=reverse, map_fn=map_fn)[:num_files]
        else:
            folders_dict[folder_key(f.name)] = sorted_paths(f, key=file_sort_key, suffix=suffix, reverse=reverse,
                                                            map_fn=lambda x:x.name)[:num_files]
    return folders_dict

# def folders_with_files(p, full_path=False, sort_key=None, suffix=None, num_files=None, folder_key=lambda x:x, make_str=False):

#     # if sort_key is None:
#     #     folders = sorted_paths(p, reverse=False)
#     # else:
#     #     folders = sorted_paths(p, key=sort_key, reverse=False)
#     folders = sorted_paths(p, key=sort_key, reverse=False)

#     folders_dict = dict()
#     for f in folders:
#          if full_path:
#                 folders_dict[folder_key(f.name)] = [str(p) for p in sorted_paths(f, key=sort_key, reverse=False)][:num_files]
#             else:
#                 folders_dict[folder_key(f.name)] = [p.name for p in sorted_paths(f, key=sort_key, reverse=False)][:num_files]
#         # if sort_key is None:
#         #     if full_path:
#         #         folders_dict[folder_key(f.name)] = [str(p) for p in sorted_paths(f, reverse=False)][:num_files]
#         #     else:
#         #         folders_dict[folder_key(f.name)] = [p.name for p in sorted_paths(f, reverse=False)][:num_files]
#         # else:
#         #     if full_path:
#         #         folders_dict[folder_key(f.name)] = [str(p) for p in sorted_paths(f, key=sort_key, reverse=False)][:num_files]    
#         #     else:
#         #         folders_dict[folder_key(f.name)] = [p.name for p in sorted_paths(f, key=sort_key, reverse=False)][:num_files]
#     return folders_dict

def end_of_path(p, n=2):
    parts = p.parts
    p = Path(parts[-n])
    for i in range(-(n-1), 0):
        p/=parts[i]
    return p

def process_landmarks(lm):
    lm_keys = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
               'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
    landmarks = []
    check = False
    if len(lm) > 0:
        lm = lm[0]
        if sum(x < 0 for x in list(sum([list(sum(x,())) for x in list(lm.values())],[]))) == 0:
            check = True
            marks = []
            for k in lm_keys:
                marks += lm[k]
            landmarks+=(list(sum(marks,())))
    return landmarks,check

def greater_2_power(x):
    return 1<<(x-1).bit_length()

def pad(img, size=(256,256,3), value=1):
    padded = np.ones(size, dtype=np.uint8)*np.uint8(value)
    pad_h, pad_w = size[:2]
    img_h, img_w = img.shape[:2]
    offset_h = (pad_h-img_h)//2
    offset_w = (pad_w-img_w)//2
    padded[offset_h:img_h+offset_h, offset_w:img_w+offset_w] = img
    return padded

def remove_pad(img, shape=(256,256)):
    img_h, img_w = img.shape[:2]
    pad_h, pad_w = (img_h-shape[0])//2, (img_w-shape[1])//2
    print(pad_h, pad_w)
    return img[pad_h:shape[0]+pad_h, pad_w:shape[1]+pad_w]

class imgs_to_batch_dataset(Dataset):
    
    def __init__(self, data, transforms_ = None, channels = 3):
        super(imgs_to_batch_dataset, self).__init__()
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.channels = channels
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        try:
            img_path = self.data.iloc[index, 0]
            if self.channels == 3:
                img = bgr2rgb(cv2.imread(str(img_path)))
            else:    
                img = cv2.imread(str(img_path),0)
        except:
            img_path = ''
            img = np.array(self.data.iloc[index, 0])
        self.tfms = albu.Compose(self.transforms_)
        x = self.tfms(image=img)['image']
        if self.channels == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        return x,img_path
    
class WeightedMultilabel(nn.Module):
    def __init__(self, weights):
        super(WeightedMultilabel,self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.weights = weights.unsqueeze(0)

    def forward(self,outputs, targets):
        loss = torch.sum(self.loss(outputs, targets) * self.weights) 
        return loss

class MultiConcatHeader(nn.Module):
    def __init__(self,fc1,fc2):
        super(MultiConcatHeader, self).__init__()
        self.fc1 = fc1
        self.fc2 = fc2
    def forward(self,x):
        single_label = self.fc1(x)
        single_index = torch.argmax(torch.softmax(single_label,1),dim=1).float().unsqueeze(1)
        # print(flatten_tensor(x).shape,single_index.shape)
        multi_input = torch.cat((flatten_tensor(x),single_index),dim=1)
        multi_label = self.fc2(multi_input)
        return single_label,multi_label

class MultiSeparateHeader(nn.Module):
    def __init__(self,fc1,fc2):
        super(MultiSeparateHeader, self).__init__()
        self.fc1 = fc1
        self.fc2 = fc2
    def forward(self,x):
        single_label = self.fc1(x)
        multi_label = self.fc2(x)
        return single_label,multi_label