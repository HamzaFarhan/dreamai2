from .utils import *
from .dai_imports import *

class DaiDataset(Dataset):
    
    def __init__(self, data, data_dir='', tfms=None, ss_tfms=None, channels=3, meta_idx=None, **kwargs):
        super(DaiDataset, self).__init__()
        self.tfms_list = []
        self.data_dir = str(data_dir)
        self.data = data
        self.tfms = tfms
        self.ss_tfms = ss_tfms
        self.meta_idx = meta_idx
        if self.tfms is not None:
            self.tfms_list.append(self.tfms)
        if self.ss_tfms is not None:
            self.tfms_list.append(self.ss_tfms)
        # if tfms is not None:
            # self.tfms = albu.Compose(tfms)
        # else:
            # self.tfms = tfms
        self.channels = channels
        for k in kwargs:
            setattr(self, k, kwargs[k])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        try:
            img_path = os.path.join(self.data_dir, self.data.iloc[index, 0])
        except:
            img_path = os.path.join(self.data_dir, self.data[index, 0])
        # if not Path(img_path).exists():
        #print(img_path)
        if self.channels == 3:
            img = rgb_read(img_path)
        else:    
            img = c1_read(img_path)
        try:
            y = self.data.iloc[index, 1]
        except:
            y = self.data[index, 1]
        if self.tfms is not None:
            x = apply_tfms(img.copy(), self.tfms)
            # x = self.tfms(image=img)['image']
            if self.channels == 1:
                x = x.unsqueeze(0)
        else:
            x = img
        if is_str(y) and hasattr(self, 'class_names'):
            y = self.class_names.index(y)
        ret = {'x':x, 'label':y, 'path':self.data.iloc[index, 0]}
        if self.ss_tfms is not None:
            # if self.ss_data is not None:
            #     try:
            #         img_path = os.path.join(self.data_dir, self.ss_data.iloc[index, 0])
            #     except:
            #         img_path = os.path.join(self.data_dir, self.ss_data[index, 0])
            #     if self.channels == 3:
            #         img = rgb_read(img_path)
            #     else:    
            #         img = c1_read(img_path)

            # x2 = apply_tfms(img, self.ss_tfms)
            # if self.channels == 1:
            #     x2 = x2.unsqueeze(0)
            index2 = random.choice(range(len(self.data)))
            try:
                img_path2 = os.path.join(self.data_dir, self.data.iloc[index2, 0])
            except:
                img_path2 = os.path.join(self.data_dir, self.data[index2, 0])
            if self.channels == 3:
                img2 = rgb_read(img_path2)
            else:    
                img2 = c1_read(img_path2)
            x2 = apply_tfms(img2.copy(), self.ss_tfms)
            if self.channels == 1:
                x2 = x2.unsqueeze(0)
            norm_t = get_norm(self.tfms)
            if norm_t:
                mean = norm_t.mean
                std = norm_t.std
            else:
                std = None
                mean = None
            resize_t = list(self.tfms)[0]
            h,w = resize_t.height, resize_t.width
            img2_tfms = instant_tfms(h, w, img_mean=mean, img_std=std)[0]
            img2 = apply_tfms(img2, img2_tfms)
            if self.channels == 1:
                img2 = img2.unsqueeze(0)
            
            ret = {'x':x, 'label':y, 'ss_img':img2, 'x2':x2, 'path':self.data.iloc[index, 0]}
        # else:
            # img2 = x
            # x2 = x
        if self.meta_idx is not None:
            if not list_or_tuple(self.meta_idx):
                meta1,meta2 = self.meta_idx, self.meta_idx+1
            else:
                meta1,meta2 = self.meta_idx
            try:
                ret['meta'] = torch.cat([tensor(m) for m in self.data.iloc[index, meta1:meta2]]).float()
            except:
                ret['meta'] = torch.cat([tensor([m]) for m in self.data.iloc[index, meta1:meta2]]).float()

        return ret
        # return {'x':x, 'label':y, 'ss_img':img2, 'x2':x2, 'path':self.data.iloc[index, 0]}
        # return x, y, img2, x2, self.data.iloc[index, 0]

    def get_at_index(self, index, denorm=True, show=True):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        if self.channels == 3:
            img = rgb_read(img_path)
        else:    
            img = c1_read(img_path)
        
        y = self.data.iloc[index, 1]
        if not is_str(y):
            label = self.data.iloc[index, 2]
        else:
            label = y
        if self.tfms is not None:
            x = apply_tfms(img.copy(), self.tfms)
            # x = (self.tfms(image=img)['image'])
            if self.channels == 1:
                x = x.unsqueeze(0)
            x = tensor_to_img(x)
            if denorm:
                norm_t = get_norm(self.tfms)
                if norm_t:
                    mean = norm_t.mean
                    std = norm_t.std
                    x = denorm_img(x, mean, std)
        else:
            x = img
        p = self.data.iloc[index, 0]

        ret = {'x':x, 'label':y, 'path':p}

        if self.ss_tfms is not None:
            # if self.ss_data is not None:
            #     try:
            #         img_path = os.path.join(self.data_dir, self.ss_data.iloc[index, 0])
            #     except:
            #         img_path = os.path.join(self.data_dir, self.ss_data[index, 0])
            #     if self.channels == 3:
            #         img = rgb_read(img_path)
            #     else:    
            #         img = c1_read(img_path)
            index2 = random.choice(range(len(self.data)))
            try:
                img_path2 = os.path.join(self.data_dir, self.data.iloc[index2, 0])
            except:
                img_path2 = os.path.join(self.data_dir, self.data[index2, 0])
            if self.channels == 3:
                img2 = rgb_read(img_path2)
            else:    
                img2 = c1_read(img_path2)
            x2 = apply_tfms(img2.copy(), self.ss_tfms)
            if self.channels == 1:
                x2 = x2.unsqueeze(0)
            x2 = tensor_to_img(x2)
            norm_t = get_norm(self.tfms)
            if norm_t:
                mean = norm_t.mean
                std = norm_t.std
            else:
                std = None
                mean = None
            resize_t = list(self.tfms)[0]
            h,w = resize_t.height, resize_t.width
            img2_tfms = instant_tfms(h, w, img_mean=mean, img_std=std)[0]
            img2 = apply_tfms(img2, img2_tfms)
            if self.channels == 1:
                img2 = img2.unsqueeze(0)
            img2 = tensor_to_img(img2)
            if denorm:
                # norm_t = get_norm(self.ss_tfms)
                # if norm_t:
                    # mean = norm_t.mean
                    # std = norm_t.std
                img2 = denorm_img(img2, mean, std)
                x2 = denorm_img(x2, mean, std)
            
            ret = {'x':x, 'label':y, 'ss_img':img2, 'x2':x2, 'path':p}
        # else:
            # x2 = x
            # img2 = x
                
        if show:
            print(f'path:{p}')
            if self.tfms is None:
                aug = ''
            else: aug = ' Augmented'
            plt_show(x, title=f'Normal{aug}: {label}')
            if self.ss_tfms is not None:
                plt_show(img2, title=f'SS Image: {label}')
                plt_show(x2, title=f'SS Augmented: {label}')

        if self.meta_idx is not None:
            if not list_or_tuple(self.meta_idx):
                meta1,meta2 = self.meta_idx, self.meta_idx+1
            else:
                meta1,meta2 = self.meta_idx
            try:
                ret['meta'] = torch.cat([tensor(m) for m in self.data.iloc[index, meta1:meta2]])
            except:
                ret['meta'] = torch.cat([tensor([m]) for m in self.data.iloc[index, meta1:meta2]])
        return ret
        # return {'x':x, 'label':y, 'ss_img':img2, 'x2':x2, 'path':p}

class SimilarityDataset(Dataset):
    
    def __init__(self, data, data_dir='', tfms=None, tfms2=None, channels=3,**kwargs):
        super(SimilarityDataset, self).__init__()
        self.tfms_list = []
        self.data_dir = str(data_dir)
        self.data = data
        self.tfms = tfms
        self.tfms2 = tfms2
        if self.tfms is not None:
            self.tfms_list.append(self.tfms)
        if self.tfms2 is not None:
            self.tfms_list.append(self.tfms2)
        # if tfms is not None:
            # self.tfms = albu.Compose(tfms)
        # else:
            # self.tfms = tfms
        self.channels = channels
        for k in kwargs:
            setattr(self, k, kwargs[k])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        try:
            img_path = os.path.join(self.data_dir, self.data.iloc[index, 0])
        except:
            img_path = os.path.join(self.data_dir, self.data[index, 0])
        # if not Path(img_path).exists():
        #print(img_path)
        if self.channels == 3:
            img = rgb_read(img_path)
        else:    
            img = c1_read(img_path)
        try:
            y = self.data.iloc[index, 1]
        except:
            y = self.data[index, 1]
        if self.tfms is not None:
            x = apply_tfms(img.copy(), self.tfms)
            # x = self.tfms(image=img)['image']
            if self.channels == 1:
                x = x.unsqueeze(0)
        else:
            x = img
        if is_str(y) and hasattr(self, 'class_names'):
            y = self.class_names.index(y)
        # ret = {'x':x, 'label':y, 'path':self.data.iloc[index, 0]}

        index2 = random.choice(range(len(self.data)))
        try:
            y2 = self.data.iloc[index2, 1]
        except:
            y2 = self.data[index2, 1]
        if is_str(y2) and hasattr(self, 'class_names'):
            y2 = self.class_names.index(y2)
        try:
            if y2 == y:
                same = 1
            else:
                same = -1
        except:
            if (y2 == y).all():
                same = 1
            else:
                same = -1

        while y2 != y:
            index2 = random.choice(range(len(self.data)))
            try:
                y2 = self.data.iloc[index2, 1]
            except:
                y2 = self.data[index2, 1]
            if is_str(y2) and hasattr(self, 'class_names'):
                y2 = self.class_names.index(y2)
        try:
            img_path2 = os.path.join(self.data_dir, self.data.iloc[index2, 0])
        except:
            img_path2 = os.path.join(self.data_dir, self.data[index2, 0])
        if self.channels == 3:
            img2 = rgb_read(img_path2)
        else:    
            img2 = c1_read(img_path2)

        if self.tfms2 is not None:
            x2 = apply_tfms(img2.copy(), self.tfms2)
            # x = self.tfms(image=img)['image']
            if self.channels == 1:
                x2 = x2.unsqueeze(0)
        elif self.tfms is not None:
            x2 = apply_tfms(img2.copy(), self.tfms)
            # x = self.tfms(image=img)['image']
            if self.channels == 1:
                x2 = x2.unsqueeze(0)
        else:
            x2 = img2
        
        ret = {'x':x, 'label':y, 'path':self.data.iloc[index, 0], 'x2':x2, 'label2':y2,
               'path2':self.data.iloc[index2, 0], 'same':same}

        return ret

    def get_at_index(self, index, denorm=True, show=True):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        if self.channels == 3:
            img = rgb_read(img_path)
        else:    
            img = c1_read(img_path)
        
        y = self.data.iloc[index, 1]
        if not is_str(y):
            label = self.data.iloc[index, 2]
        else:
            label = y
        if self.tfms is not None:
            x = apply_tfms(img.copy(), self.tfms)
            # x = (self.tfms(image=img)['image'])
            if self.channels == 1:
                x = x.unsqueeze(0)
            x = tensor_to_img(x)
            if denorm:
                norm_t = get_norm(self.tfms)
                if norm_t:
                    mean = norm_t.mean
                    std = norm_t.std
                    x = denorm_img(x, mean, std)
        else:
            x = img
        p = self.data.iloc[index, 0]

        ret = {'x':x, 'label':y, 'path':p}

        index2 = random.choice(range(len(self.data)))
        try:
            y2 = self.data.iloc[index2, 1]
        except:
            y2 = self.data[index2, 1]

        # while y2 != y:
        #     index2 = random.choice(range(len(self.data)))
        #     try:
        #         y2 = self.data.iloc[index2, 1]
        #     except:
        #         y2 = self.data[index2, 1]
        
        if not is_str(y2):
            label2 = self.data.iloc[index2, 2]
        else:
            label2 = y2        

        try:
            img_path2 = os.path.join(self.data_dir, self.data.iloc[index2, 0])
        except:
            img_path2 = os.path.join(self.data_dir, self.data[index2, 0])
        if self.channels == 3:
            img2 = rgb_read(img_path2)
        else:    
            img2 = c1_read(img_path2)
        
        if self.tfms2 is not None:
            x2 = apply_tfms(img2.copy(), self.tfms2)
            # x = self.tfms(image=img)['image']
            if self.channels == 1:
                x2 = x2.unsqueeze(0)
            x2 = tensor_to_img(x2)
            norm_t = get_norm(self.tfms2)
            if norm_t:
                mean = norm_t.mean
                std = norm_t.std
            else:
                std = None
                mean = None
            resize_t = list(self.tfms2)[0]
            h,w = resize_t.height, resize_t.width
            img2_tfms = instant_tfms(h, w, img_mean=mean, img_std=std)[0]
            img2 = apply_tfms(img2, img2_tfms)
            if self.channels == 1:
                img2 = img2.unsqueeze(0)
            img2 = tensor_to_img(img2)
            if denorm:
                # norm_t = get_norm(self.ss_tfms)
                # if norm_t:
                    # mean = norm_t.mean
                    # std = norm_t.std
                img2 = denorm_img(img2, mean, std)
                x2 = denorm_img(x2, mean, std)
        elif self.tfms is not None:
            x2 = apply_tfms(img2.copy(), self.tfms)
            # x = self.tfms(image=img)['image']
            if self.channels == 1:
                x2 = x2.unsqueeze(0)
            x2 = tensor_to_img(x2)
            norm_t = get_norm(self.tfms)
            if norm_t:
                mean = norm_t.mean
                std = norm_t.std
            else:
                std = None
                mean = None
            resize_t = list(self.tfms)[0]
            h,w = resize_t.height, resize_t.width
            img2_tfms = instant_tfms(h, w, img_mean=mean, img_std=std)[0]
            img2 = apply_tfms(img2, img2_tfms)
            if self.channels == 1:
                img2 = img2.unsqueeze(0)
            img2 = tensor_to_img(img2)
            if denorm:
                # norm_t = get_norm(self.ss_tfms)
                # if norm_t:
                    # mean = norm_t.mean
                    # std = norm_t.std
                img2 = denorm_img(img2, mean, std)
                x2 = denorm_img(x2, mean, std)
        else:
            x2 = img2

        p2 = self.data.iloc[index2, 0]

        ret = {'x':x, 'label':y, 'img2':img2, 'x2':x2, 'label2':y2, 'path':p, 'path2':p2}
                
        if show:
            print(f'path1: {p}')
            # if self.tfms is None:
                # aug = ''
            # else: aug = ' Augmented'
            plt_show(x, title=f'Image1: {label}')
            # if self.ss_tfms is not None:
            # plt_show(img2, title=f'SS Image: {label}')
            print(f'path2: {p2}')
            plt_show(x2, title=f'Image2: {label2}')

        return ret
        # return {'x':x, 'label':y, 'ss_img':img2, 'x2':x2, 'path':p}

class PredDataset(Dataset):
    def __init__(self, data, data_dir='', tfms=None, channels=3, **kwargs):
        super(PredDataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.tfms = tfms
        self.channels = 3
        for k in kwargs:
            setattr(self, k, kwargs[k])
    
    def __getitem__(self, index):

        try:
            img_path = os.path.join(self.data_dir, self.data[index])
            if self.channels == 3:
                img = rgb_read(img_path)
            else:    
                img = c1_read(img_path)
        except:
            img = self.data[index]

        if self.tfms is not None:
            x = self.tfms(image=img)['image']
            if self.channels == 1:
                x = x.unsqueeze(0)
        else:
            x = img
        return x

    def __len__(self): return len(self.data)

class dai_classifier_dataset(Dataset):
    
    def __init__(self, data, data_dir='', tfms=instant_tfms(), channels=3, class_names=[], **kwargs):
        super(dai_classifier_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.tfms = albu.Compose(tfms)
        self.channels = channels
        self.class_names = class_names
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        if self.channels == 3:
            img = rgb_read(img_path)
        else:    
            img = c1_read(img_path)
            
        y = self.data.iloc[index, 1]    
        x = self.tfms(image=img)['image']
        if self.channels == 1:
            x = x.unsqueeze(0)
        if is_str(y):
            y = self.class_names.index(y)
        return x, y, self.data.iloc[index, 0]

def get_classifier_dls(df, val_df=None, test_df=None, data_dir='', dset=DaiDataset,
                       tfms=instant_tfms(224, 224), ss_tfms=None, bs=64, shuffle=True,
                       pin_memory=True, num_workers=4, force_one_hot=False, meta_idx=None,
                       class_names=None, split=True, val_size=0.2, test_size=0.15, **kwargs):
    # if len(kwargs) > 0:
        # dset = partial(dset, **kwargs)
    # if list_or_tuple(dfs):
    #     df = dfs[0]
    #     if len(dfs) > 1:
    #         other_dfs = dfs[1:]
    #         # ss_df = dfs[-1]
    # else:
    #     df = dfs
    #     other_dfs = []
    #     # ss_df = None
    def df_one_hot(df):
        df = df.copy()
        labels = list_map(df.iloc[:,1], lambda x:str(x).split())
        one_hot_labels = dai_one_hot(labels, class_names)
        df['one_hot'] = list(one_hot_labels)
        cols = df.columns.to_list()
        df = df[[cols[0], cols[-1], *cols[1:-1]]]
        return df

    labels = list_map(df.iloc[:,1], lambda x:str(x).split())
    is_multi = np.array(list_map(labels, lambda x:len(x)>1)).any()
    if class_names is None:
        class_names = np.unique(flatten_list(labels))
    class_names = list_map(class_names, str)
    stratify_idx = 1
    # ss_transforms = [ss_tfms[0]]
    if is_multi or force_one_hot:
        # one_hot_labels = dai_one_hot(labels, class_names)    
        dfs = [df, val_df, test_df]
        for i in range(3):
            if dfs[i] is not None:
                dfs[i] = df_one_hot(dfs[i])
        df, val_df, test_df = dfs
        if meta_idx is not None:
            if not list_or_tuple(meta_idx):
                meta_idx+=1
            else:
                meta_idx = [m_id+1 for m_id in meta_idx]
        # df['one_hot'] = list(one_hot_labels)
        # cols = df.columns.to_list()
        # df = df[[cols[0], cols[-1], *cols[1:-1]]]
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
    if ss_tfms is not None:
        if list_or_tuple(ss_tfms): ss_tfms = ss_tfms[0]
    dsets = [dset(data_dir=data_dir, data=df, tfms=tfms_, **kwargs,
                  ss_tfms=ss_tfms, class_names=class_names, meta_idx=meta_idx) for df,tfms_ in zip(dfs, transforms_)]
    dls = get_dls(dsets=[dsets[0]], bs=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if split:
        dls += get_dls(dsets=dsets[1:], bs=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dls = DataLoaders(*dls)
    dls.class_names = class_names
    dls.num_classes = len(class_names)
    dls.class_weights = [get_class_weights(df) for df in dfs]
    dls.is_multi = is_multi
    dls.data_type = 'classification'
    dls.is_one_hot = is_multi + force_one_hot
    if dls.is_one_hot:
        dls.suggested_crit = nn.BCEWithLogitsLoss()
        dls.suggested_metric = 'multi_accuracy'
    else:
        dls.suggested_crit = nn.CrossEntropyLoss()
        dls.suggested_metric = 'accuracy'

    return dls

def get_similarity_dls(df, val_df=None, test_df=None, data_dir='', dset=SimilarityDataset,
                       tfms=instant_tfms(224, 224), tfms2=None, bs=64, shuffle=True,
                       pin_memory=True, num_workers=4, force_one_hot=False,
                       class_names=None, split=True, val_size=0.2, test_size=0.15, **kwargs):
    # if len(kwargs) > 0:
        # dset = partial(dset, **kwargs)
    # if list_or_tuple(dfs):
    #     df = dfs[0]
    #     if len(dfs) > 1:
    #         other_dfs = dfs[1:]
    #         # ss_df = dfs[-1]
    # else:
    #     df = dfs
    #     other_dfs = []
    #     # ss_df = None
    def df_one_hot(df):
        df = df.copy()
        labels = list_map(df.iloc[:,1], lambda x:str(x).split())
        one_hot_labels = dai_one_hot(labels, class_names)
        df['one_hot'] = list(one_hot_labels)
        cols = df.columns.to_list()
        df = df[[cols[0], cols[-1], *cols[1:-1]]]
        return df

    labels = list_map(df.iloc[:,1], lambda x:str(x).split())
    is_multi = np.array(list_map(labels, lambda x:len(x)>1)).any()
    if class_names is None:
        class_names = np.unique(flatten_list(labels))
    class_names = list_map(class_names, str)
    stratify_idx = 1
    # ss_transforms = [ss_tfms[0]]
    if is_multi or force_one_hot:
        # one_hot_labels = dai_one_hot(labels, class_names)    
        dfs = [df, val_df, test_df]
        for i in range(3):
            if dfs[i] is not None:
                dfs[i] = df_one_hot(dfs[i])
        df, val_df, test_df = dfs
        # df['one_hot'] = list(one_hot_labels)
        # cols = df.columns.to_list()
        # df = df[[cols[0], cols[-1], *cols[1:-1]]]
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
    dsets = [dset(data_dir=data_dir, data=df, tfms=tfms_, **kwargs,
                  class_names=class_names) for df,tfms_ in zip(dfs, transforms_)]
    dls = get_dls(dsets=[dsets[0]], bs=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if split:
        dls += get_dls(dsets=dsets[1:], bs=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dls = DataLoaders(*dls)
    dls.class_names = class_names
    dls.num_classes = len(class_names)
    dls.is_multi = is_multi
    dls.data_type = 'similarity'
    dls.is_one_hot = is_multi + force_one_hot
    dls.suggested_crit = nn.CosineEmbeddingLoss()
    dls.suggested_metric = 'loss'

    return dls

class DataLoaders():
    def __init__(self, train=None, valid=None, test=None):
        store_attr(self, 'train,valid,test')
        if self.train:
            self.train_ds = self.train.dataset
            self.train_dl = partial(DataLoader, dataset=self.train_ds, batch_size=self.train.batch_size,
                                    shuffle=is_shuffle(self.train), num_workers=self.train.num_workers,
                                    pin_memory=self.train.pin_memory)
        if self.valid:
            self.valid_ds = self.valid.dataset
            self.valid_dl = partial(DataLoader, dataset=self.valid_ds, batch_size=self.valid.batch_size,
                                    shuffle=is_shuffle(self.valid), num_workers=self.valid.num_workers,
                                    pin_memory=self.valid.pin_memory)
        if self.test:
            self.test_ds = self.test.dataset
            self.test_dl = partial(DataLoader, dataset=self.test_ds, batch_size=self.test.batch_size,
                                   shuffle=is_shuffle(self.test), num_workers=self.test.num_workers,
                                   pin_memory=self.test.pin_memory)

        norm_t = get_norm(self.train_ds.tfms)
        if norm_t:
            self.normalize = partial(apply_tfms, tfms=norm_t)
            self.denorm = partial(denorm_img, mean=norm_t.mean, std=norm_t.std)
            self.img_mean = norm_t.mean
            self.img_std = norm_t.std
        else:
            self.normalize = noop
            self.denorm = noop
            self.img_mean, self.img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.suggested_crit = nn.CrossEntropyLoss()
        self.suggested_metric = 'loss'

    def one_batch(self, dl_name='train'):
        return next_batch(getattr(self, dl_name))

    def assign_dls(self, train=None, valid=None, test=None):
        store_attr(self, 'train,valid,test')
        if self.train:
            self.train_ds = self.train.dataset
            self.train_dl = partial(DataLoader, dataset=self.train_ds, batch_size=self.train.batch_size,
                                    shuffle=is_shuffle(self.train), num_workers=self.train.num_workers,
                                    pin_memory=self.train.pin_memory)
        if self.valid:
            self.valid_ds = self.valid.dataset
            self.valid_dl = partial(DataLoader, dataset=self.valid_ds, batch_size=self.valid.batch_size,
                                    shuffle=is_shuffle(self.valid), num_workers=self.valid.num_workers,
                                    pin_memory=self.valid.pin_memory)
        if self.test:
            self.test_ds = self.test.dataset
            self.test_dl = partial(DataLoader, dataset=self.test_ds, batch_size=self.test.batch_size,
                                   shuffle=is_shuffle(self.test), num_workers=self.test.num_workers,
                                   pin_memory=self.test.pin_memory)

    def progressive_resize(self, h=224, w=224, bs=32):
        new_dls = []
        for dtype in ['train', 'valid', 'test']:
            if getattr(self, dtype):
                ds = getattr(self, f'{dtype}_ds')
                if hasattr(ds, 'tfms_list'):
                    if len(ds.tfms_list) > 0:
                        for t in ds.tfms_list:
                            set_resize_dims(t, h=h, w=w)
                elif hasattr(ds, 'tfms'):
                    set_resize_dims(ds.tfms, h=h, w=w)
                new_dls.append(new_dl(getattr(self, f'{dtype}_dl'), bs=bs))
        self.assign_dls(*new_dls)

    # def progressive_resize(self, h=224, w=224, bs=32):
    #     new_train_dl, new_valid_dl, new_test_dl = None, None, None

    #     if self.train:
    #         # set_resize_dims(self.train_ds.tfms, h=h, w=w)
    #         # if self.train_ds.ss_tfms is not None:
    #         if len(self.train_ds.tfms_list) > 0:
    #             for t in self.train_ds.tfms_list:
    #                 # set_resize_dims(self.train_ds.ss_tfms, h=h, w=w)
    #                 set_resize_dims(t, h=h, w=w)
    #         new_train_dl = new_dl(self.train_dl, bs=bs)
    #     if self.valid:
    #         set_resize_dims(self.valid_ds.tfms, h=h, w=w)
    #         if self.valid_ds.ss_tfms is not None:
    #             set_resize_dims(self.valid_ds.ss_tfms, h=h, w=w)
    #         new_valid_dl = new_dl(self.valid_dl, bs=bs)
    #     if self.test:
    #         set_resize_dims(self.test_ds.tfms, h=h, w=w)
    #         if self.test_ds.ss_tfms is not None:
    #             set_resize_dims(self.test_ds.ss_tfms, h=h, w=w)
    #         new_test_dl = new_dl(self.test_dl, bs=bs)

    #     self.assign_dls(new_train_dl, new_valid_dl, new_test_dl)

    def __len__(self):
        return sum([(self.train!=None) + (self.valid!=None) + (self.test!=None)])

def get_dls(dsets, bs=32, shuffle=True, num_workers=4, pin_memory=True):
    dls = [DataLoader(dset, batch_size=bs, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory) for dset in dsets]
    # for dl in dls:
        # dl.suggested_metric = 'loss'
        # dl.suggested_crit = None
    return dls

def get_class_weights(df):
    df = df.copy()
    try:
        counts = sum(df.iloc[:,1])
        total = sum(counts)
        w = [(x/total) for x in counts]
    except:
        w = list(df.iloc[:,1].value_counts(normalize=True).sort_index())    
    return 1-tensor(w)

def get_data_stats(df, data_dir='', image_size=224, stats_percentage=0.7, bs=32, device=None):
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print('Calculating dataset mean and std. This may take a while.', end='')
    print('Calculating dataset mean and std. This may take a while.\n')
    frac_data = df.copy().sample(frac=stats_percentage).reset_index(drop=True).copy()
    tfms = instant_tfms(image_size, image_size)[1]
    dset = DaiDataset(frac_data, data_dir=data_dir, tfms=tfms)
    dl = DataLoader(dset, batch_size=bs, num_workers=4)
    batches = len(dl)
    # print('.', end='')
    mean = 0.0
    print('Mean loop:')
    for i,data_batch in enumerate(dl):
        try:
            if i % (batches//10) == 0:
                print(f'Batch: {i+1}/{batches}')
        except:
            print(f'Batch: {i+1}/{batches}')
        # images = data_batch[0]
        images = data_batch['x'].to(device)
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)
    # print('.', end='')
    var = 0.0
    print('\nStd loop:')
    for i,data_batch in enumerate(dl):
        try:
            if i % (batches//10) == 0:
                print(f'Batch: {i+1}/{batches}')
        except:
            print(f'Batch: {i+1}/{batches}')
        # images = data_batch[0]
        images = data_batch['x'].to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(dl.dataset)*image_size*image_size))
    print('\nDone.')
    return mean.cpu(), std.cpu()



































class dai_image_csv_dataset(Dataset):
    
    def __init__(self, data_dir, data, transforms_ = None, obj = False, seg = False,
                    minorities = None, diffs = None, bal_tfms = None, channels = 3, **kwargs):
        super(dai_image_csv_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.obj = obj
        self.seg = seg
        self.minorities = minorities
        self.diffs = diffs
        self.bal_tfms = bal_tfms
        self.channels = channels
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        if self.channels == 3:
            img = utils.bgr2rgb(cv2.imread(str(img_path)))
        else:    
            img = cv2.imread(str(img_path),0)

        # img = Image.open(img_path)
        # if self.channels == 3:
        #     img = img.convert('RGB')
        # else:    
        #     img = img.convert('L')

        y = self.data.iloc[index, 1]    
        if self.minorities and self.bal_tfms:
            if y in self.minorities:
                if hasattr(self.bal_tfms,'transforms'):
                    for tr in self.bal_tfms.transforms:
                        tr.p = self.diffs[y]
                    l = [self.bal_tfms]
                    l.extend(self.transforms_)
                    self.tfms = albu.Compose(l)    
                else:            
                    for t in self.bal_tfms:
                        t.p = self.diffs[y]
                    self.transforms_[1:1] = self.bal_tfms    
                    # self.tfms = transforms.Compose(self.transforms_)
                    self.tfms = albu.Compose(self.transforms_)
                    # print(self.tfms)
            else:
                # self.tfms = transforms.Compose(self.transforms_)
                self.tfms = albu.Compose(self.transforms_)
        else:    
            # self.tfms = transforms.Compose(self.transforms_)
            self.tfms = albu.Compose(self.transforms_)
        # x = self.tfms(img)
        x = self.tfms(image=img)['image']
        if self.channels == 1:
            x = x.unsqueeze(0)
        if self.seg:
            mask = Image.open(self.data.iloc[index, 1])
            seg_tfms = albu.Compose([self.tfms.transforms[0]])
            y = torch.from_numpy(np.array(seg_tfms(mask))).long().squeeze(0)

        # if self.obj:
        #     s = x.size()[1]
        #     if isinstance(s,tuple):
        #         s = s[0]
        #     row_scale = s/img.size[0]
        #     col_scale = s/img.size[1]
        #     y = rescale_bbox(y,row_scale,col_scale)
        #     y.squeeze_()
        #     y2 = self.data.iloc[index, 2]
        #     y = (y,y2)
        return (x,y,self.data.iloc[index, 0])

class dai_obj_dataset(Dataset):

    def __init__(self, data_dir, data, tfms, has_difficult=False):
        super(dai_obj_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.tfms = tfms
        self.has_difficult = has_difficult

        assert tfms is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        try:
            boxes = torch.FloatTensor(literal_eval(self.data.iloc[index,1]))
            labels = torch.LongTensor(literal_eval(self.data.iloc[index,2]))
            if self.has_difficult:
                difficulties = torch.ByteTensor(literal_eval(self.data.iloc[index,3]))
            else:
                difficulties = None
        except:        
            boxes = torch.FloatTensor(self.data.iloc[index,1])
            labels = torch.LongTensor(self.data.iloc[index,2])
            if self.has_difficult:
                difficulties = torch.ByteTensor(self.data.iloc[index,3])
            else:
                difficulties = None

        # Apply transformations
        image, boxes, labels, difficulties = self.tfms(image, boxes, labels, difficulties)

        return image, boxes, labels, difficulties

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

class dai_image_csv_dataset_food(Dataset):
    
    def __init__(self, data_dir, data, transforms_ = None):
        super(dai_image_csv_dataset_food, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img = Image.open(img_path)
        img = img.convert('RGB')
        y1,y2 = self.data.iloc[index, 1],self.data.iloc[index, 2]    
        self.tfms = transforms.Compose(self.transforms_)    
        x = self.tfms(img)
        return (x,y1,y2)

class dai_image_csv_dataset_multi_head(Dataset):

    def __init__(self, data_dir, data, transforms_ = None, channels=3):
        super(dai_image_csv_dataset_multi_head, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.channels = channels
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        if self.channels == 3:
            img = utils.bgr2rgb(cv2.imread(str(img_path)))
        else:    
            img = cv2.imread(str(img_path),0)

        # img = Image.open(img_path)
        # if self.channels == 3:
        #     img = img.convert('RGB')
        # else:    
        #     img = img.convert('L')

        y1,y2 = self.data.iloc[index, 1],self.data.iloc[index, 2]    
        self.tfms = albu.Compose(self.transforms_)    
        x = self.tfms(image=img)['image'].unsqueeze(0)
        # self.tfms = transforms.Compose(self.transforms_)    
        # x = self.tfms(img)
        return (x,y1,y2)

class dai_image_csv_dataset_landmarks(Dataset):

    def __init__(self, data_dir, data, transforms_ = None, obj = False,
                    minorities = None, diffs = None, bal_tfms = None,channels=3):
        super(dai_image_csv_dataset_landmarks, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.obj = obj
        self.minorities = minorities
        self.diffs = diffs
        self.bal_tfms = bal_tfms
        self.channels = channels
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img = Image.open(img_path)
        if self.channels == 3:
            img = img.convert('RGB')
        else:    
            img = img.convert('L')
        y1,y2 = self.data.iloc[index, 1],self.data.iloc[index, 2]
        try:
            y2 = torch.Tensor(literal_eval(y2))
        except:
            y2 = torch.Tensor(y2)
        self.tfms = transforms.Compose(self.transforms_)    
        x = self.tfms(img)
        s = x.shape[1]
        if isinstance(s,tuple):
            s = s[0]
        row_scale = s/img.size[1]
        col_scale = s/img.size[0]
        y2 = ((rescale_landmarks(copy.deepcopy(y2),row_scale,col_scale).squeeze())-s)/s
        return (x,y1,y2)

class dai_image_dataset(Dataset):

    def __init__(self, data_dir, data, input_transforms=None, target_transforms=None, **kwargs):
        super(dai_image_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.input_transforms = None
        self.target_transforms = None
        if input_transforms:
            self.input_transforms = albu.Compose(input_transforms)
        if target_transforms:    
            self.target_transforms = albu.Compose(target_transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img = utils.bgr2rgb(cv2.imread(str(img_path)))
        img_path_2 = os.path.join(self.data_dir,self.data.iloc[index, 1])
        target = utils.bgr2rgb(cv2.imread(str(img_path_2)))
        # try:
        #     img_path_2 = os.path.join(self.data_dir,self.data.iloc[index, 1])
        #     target = utils.bgr2rgb(cv2.imread(str(img_path_2)))
        # except:
        #     print('nooo')
        #     target = utils.bgr2rgb(cv2.imread(str(img_path)))
        if self.input_transforms:
            img = self.input_transforms(image=img)['image']
        if self.target_transforms:
            target = self.target_transforms(image=target)['image']
        return img, target

class dai_super_res_dataset(Dataset):

    def __init__(self, data_dir, data, transforms_, **kwargs):
        super(dai_super_res_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.pre_transforms = albu.Compose(transforms_['pre_transforms'])
        self.pre_input_transforms = albu.Compose(transforms_['pre_input_transforms'])
        # self.downscale_transforms = albu.Compose(transforms_['downscale'])
        self.input_transforms = albu.Compose(transforms_['input'])
        self.target_transforms = albu.Compose(transforms_['target'])
        self.resized_target_transforms = albu.Compose(transforms_['resized_target'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        try:
            img_ = utils.bgr2rgb(cv2.imread(str(img_path)))
        except:
            print(img_path)
        if len(self.pre_transforms.transforms.transforms) > 0:
            img_ = self.pre_transforms(image=img_)['image']    
        target = self.target_transforms(image=img_)['image']
        if len(self.pre_input_transforms.transforms.transforms) > 0:
            img_ = self.pre_input_transforms(image=img_)['image']   
        # img_ = self.downscale_transforms(image=img_)['image']
        img = self.input_transforms(image=img_)['image']
        resized_target = self.resized_target_transforms(image=img_)['image']
        return img, target, resized_target

class dai_video_dataset(Dataset):
    def __init__(self, data, tfms):
        self.data = data
        self.tfms = tfms
        
    def __getitem__(self, index):
        
        chunk = (self.data.iloc[index, 0].split())
        label = (literal_eval(self.data.iloc[index, 1]))
        for i,c in enumerate(chunk):
            img = Image.open(c)
            chunk[i] = self.tfms(img)
        chunk = torch.stack(chunk, 0).permute(1, 0, 2, 3)
        return chunk, label
    
    def __len__(self): return len(self.data)

def rescale_landmarks(landmarks,row_scale,col_scale):
    landmarks2 = copy.deepcopy(torch.Tensor(landmarks).reshape((-1,2)))
    for lm in landmarks2:
        c,r = lm
        lm[0] = c*col_scale
        lm[1] = r*row_scale
        # lm[0] = c*row_scale
        # lm[1] = r*col_scale
    landmarks2 = landmarks2.reshape((1,-1))        
    return landmarks2

def csv_from_path(path):

    path = Path(path)
    labels_paths = list(path.iterdir())
    tr_images = []
    tr_labels = []
    for l in labels_paths:
        if l.is_dir():
            for i in list(l.iterdir()):
                if i.suffix in IMG_EXTENSIONS:
                    name = i.name
                    label = l.name
                    new_name = f'{path.name}/{label}/{name}'
                    tr_images.append(new_name)
                    tr_labels.append(label)
    if len(tr_labels) == 0:
        return None
    tr_img_label = {'Img':tr_images, 'Label': tr_labels}
    csv = pd.DataFrame(tr_img_label,columns=['Img','Label'])
    csv = csv.sample(frac=1).reset_index(drop=True)
    return csv

def add_extension(a,e):
    a = [x+e for x in a]
    return a

class DataProcessor:
    
    def __init__(self, data_path=None, train_csv=None, val_csv=None, test_csv=None,
                 tr_name='train', val_name='val', test_name='test',
                 class_names=[], extension=None, setup_data=True, **kwargs):

        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        (self.data_path,self.train_csv,self.val_csv,self.test_csv,
         self.tr_name,self.val_name,self.test_name,self.extension) = (data_path,train_csv,val_csv,test_csv,
                                                                      tr_name,val_name,test_name,extension)

        data_type = {'seg':False, 'obj':False, 'sr':False, 'enhance':False,
                          'multi_head':False, 'multi_label':False, 'single_label':False}
        self.data_type = dict(data_type, **kwargs)
        self.seg = self.data_type['seg']
        self.obj = self.data_type['obj']
        self.sr = self.data_type['sr']
        self.enhance = self.data_type['enhance']
        self.multi_head = self.data_type['multi_head']
        self.multi_label = self.data_type['multi_label']
        self.single_label = self.data_type['single_label']
        self.img_mean = self.img_std = None
        self.data_dir,self.num_classes,self.class_names = data_path,len(class_names),class_names
        if setup_data:
            self.set_up_data()
                
    def set_up_data(self,split_size = 0.15):

        (data_path,train_csv,val_csv,test_csv,tr_name,val_name,test_name) = (self.data_path,self.train_csv,self.val_csv,self.test_csv,
                                                                             self.tr_name,self.val_name,self.test_name)

        # check if paths given and also set paths
        
        if not data_path:
            data_path = os.getcwd() + '/'
            self.data_dir = data_path
        tr_path = os.path.join(data_path,tr_name)
        val_path = os.path.join(data_path,val_name)
        test_path = os.path.join(data_path,test_name)
        os.makedirs('mlflow_saved_training_models',exist_ok=True)
        if train_csv is None:
            # if (os.path.exists(os.path.join(data_path,tr_name+'.csv'))):
            #     train_csv = tr_name+'.csv'
            #     if os.path.exists(os.path.join(data_path,val_name+'.csv')):
            #         val_csv = val_name+'.csv'
            #     if os.path.exists(os.path.join(data_path,test_name+'.csv')):
            #         test_csv = test_name+'.csv'
            # else:
            train_csv,val_csv,test_csv = self.data_from_paths_to_csv(data_path,tr_path,val_path,test_path)
        # else:
        #     self.data_dir = tr_path

        train_csv_path = os.path.join(data_path,train_csv)
        train_df = pd.read_csv(train_csv_path)
        if 'Unnamed: 0' in train_df.columns:
            train_df = train_df.drop('Unnamed: 0', 1)
        img_names = [str(x) for x in list(train_df.iloc[:,0])]
        if self.extension:
            img_names = add_extension(img_names,self.extension)
        if val_csv is not None:
            val_csv_path = os.path.join(data_path,val_csv)
            val_df = pd.read_csv(val_csv_path)
            val_targets = list(val_df.iloc[:,1].apply(lambda x: str(x)))
        if test_csv is not None:
            test_csv_path = os.path.join(data_path,test_csv)
            test_df = pd.read_csv(test_csv_path)
            test_targets = list(test_df.iloc[:,1].apply(lambda x: str(x)))
        if self.seg:
            print('\nSemantic Segmentation\n')
        elif self.obj:
            print('\nObject Detection\n')
        elif self.sr:
            print('\nSuper Resolution\n')
        elif self.enhance:
            print('\nImage Enhancement\n')
        else:
            if self.multi_head:
                print('\nMulti-head Classification\n')

                train_df.fillna('',inplace=True)
                train_df_single = train_df[[train_df.columns[0],train_df.columns[1]]].copy() 
                train_df_multi = train_df[[train_df.columns[0],train_df.columns[2]]].copy()
                
                targets = list(train_df_multi.iloc[:,1].apply(lambda x: str(x)))
                lengths = [len(t) for t in [s.split() for s in targets]]
                split_targets = [t.split() for t in targets]
                try:
                    split_targets = [list(map(int,x)) for x in split_targets]
                except:
                    pass
                dai_onehot,onehot_classes = one_hot(split_targets,multi=True)
                train_df_multi.iloc[:,1] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]
                self.num_multi_classes,self.multi_class_names = len(onehot_classes),onehot_classes

                targets = list(train_df_single.iloc[:,1].apply(lambda x: str(x)))
                lengths = [len(t) for t in [s.split() for s in targets]]
                split_targets = [t.split() for t in targets]
                unique_targets = list(np.unique(targets))
                try:
                    unique_targets.sort(key=int)
                except:
                    unique_targets.sort()
                unique_targets_dict = {k:v for v,k in enumerate(unique_targets)}
                train_df_single.iloc[:,1] = pd.Series(targets).apply(lambda x: unique_targets_dict[x])
                self.num_classes,self.class_names = len(unique_targets),unique_targets

                train_df = pd.merge(train_df_single,train_df_multi,on=train_df_single.columns[0])

            elif self.multi_label:
                print('\nMulti-label Classification\n')

                train_df_concat = train_df.copy()
                if val_csv:
                    train_df_concat  = pd.concat([train_df_concat,val_df]).reset_index(drop=True,inplace=False)
                if test_csv:
                    train_df_concat  = pd.concat([train_df_concat,test_df]).reset_index(drop=True,inplace=False)

                train_df_concat.fillna('',inplace=True)
                targets = list(train_df_concat.iloc[:,1].apply(lambda x: str(x)))
                lengths = [len(t) for t in [s.split() for s in targets]]
                split_targets = [t.split() for t in targets]
                try:
                    split_targets = [list(map(int,x)) for x in split_targets]
                except:
                    pass
                dai_onehot,onehot_classes = one_hot(split_targets,self.multi_label)
                train_df_concat.iloc[:,1] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]
                train_df = train_df_concat.loc[:len(train_df)-1].copy()
                if val_csv:
                    val_df = train_df_concat.loc[len(train_df):len(train_df)+len(val_df)-1].copy().reset_index(drop=True)
                if test_csv:
                    test_df = train_df_concat.loc[len(val_df)+len(train_df):len(val_df)+len(train_df)+len(test_df)-1].copy().reset_index(drop=True)
                self.num_classes,self.class_names = len(onehot_classes),onehot_classes

                # train_df.fillna('',inplace=True)
                # targets = list(train_df.iloc[:,1].apply(lambda x: str(x)))
                # lengths = [len(t) for t in [s.split() for s in targets]]
                # split_targets = [t.split() for t in targets]
                # try:
                #     split_targets = [list(map(int,x)) for x in split_targets]
                # except:
                #     pass
                # dai_onehot,onehot_classes = one_hot(split_targets,self.multi_label)
                # train_df.iloc[:,1] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]

            else:
                print('\nSingle-label Classification\n')

                targets = list(train_df.iloc[:,1].apply(lambda x: str(x)))
                lengths = [len(t) for t in [s.split() for s in targets]]
                split_targets = [t.split() for t in targets]                
                self.single_label = True
                unique_targets = list(np.unique(targets))
                try:
                    unique_targets.sort(key=int)
                except:
                    unique_targets.sort()
                unique_targets_dict = {k:v for v,k in enumerate(unique_targets)}
                train_df.iloc[:,1] = pd.Series(targets).apply(lambda x: unique_targets_dict[x])
                if val_csv:
                    val_df.iloc[:,1] = pd.Series(val_targets).apply(lambda x: unique_targets_dict[x])
                if test_csv:
                    test_df.iloc[:,1] = pd.Series(test_targets).apply(lambda x: unique_targets_dict[x])   
                self.num_classes,self.class_names = len(unique_targets),unique_targets

        if not val_csv:
            train_df,val_df = split_df(train_df,split_size)
        if not test_csv:    
            val_df,test_df = split_df(val_df,split_size)
        tr_images = [str(x) for x in list(train_df.iloc[:,0])]
        val_images = [str(x) for x in list(val_df.iloc[:,0])]
        test_images = [str(x) for x in list(test_df.iloc[:,0])]
        if self.extension:
            tr_images = add_extension(tr_images,self.extension)
            val_images = add_extension(val_images,self.extension)
            test_images = add_extension(test_images,self.extension)
        train_df.iloc[:,0] = tr_images
        val_df.iloc[:,0] = val_images
        test_df.iloc[:,0] = test_images
        if self.single_label:
            dai_df = pd.concat([train_df,val_df,test_df]).reset_index(drop=True,inplace=False)
            dai_df.iloc[:,1] = [self.class_names[x] for x in dai_df.iloc[:,1]]
            # train_df.iloc[:,1] = [self.class_names[x] for x in train_df.iloc[:,1]]
            # val_df.iloc[:,1] = [self.class_names[x] for x in val_df.iloc[:,1]]
            # test_df.iloc[:,1] = [self.class_names[x] for x in test_df.iloc[:,1]]
            dai_df.to_csv(os.path.join(data_path,'dai_processed_df.csv'),index=False)
        train_df.to_csv(os.path.join(data_path,'dai_{}.csv'.format(self.tr_name)),index=False)
        val_df.to_csv(os.path.join(data_path,'dai_{}.csv'.format(self.val_name)),index=False)
        test_df.to_csv(os.path.join(data_path,'dai_{}.csv'.format(self.test_name)),index=False)
        self.minorities,self.class_diffs = None,None
        if self.single_label:
            self.minorities,self.class_diffs = get_minorities(train_df)
        self.data_dfs = {self.tr_name:train_df, self.val_name:val_df, self.test_name:test_df}
        data_dict = dict({'data_dfs':self.data_dfs, 'data_dir':self.data_dir,
                          'num_classes':self.num_classes, 'class_names':self.class_names},
                          **self.data_type)
        self.data_dict = data_dict
        return data_dict

    def data_from_paths_to_csv(self,data_path,tr_path,val_path = None,test_path = None):
            
        train_df = csv_from_path(tr_path)
        train_df.to_csv(os.path.join(data_path,f'dai_{self.tr_name}.csv'),index=False)
        ret = (f'dai_{self.tr_name}.csv',None,None)
        if val_path is not None:
            if os.path.exists(val_path):
                val_df = csv_from_path(val_path)
                if val_df is not None:
                    val_df.to_csv(os.path.join(data_path,f'dai_{self.val_name}.csv'),index=False)
                    ret = (f'dai_{self.tr_name}.csv',f'dai_{self.val_name}.csv',None)
        if test_path is not None:
            if os.path.exists(test_path):
                test_df = csv_from_path(test_path)
                if test_df is not None:
                    test_df.to_csv(os.path.join(data_path,f'dai_{self.test_name}.csv'),index=False)
                    ret = (f'dai_{self.tr_name}.csv',f'dai_{self.val_name}.csv',f'dai_{self.test_name}.csv')        
        return ret
        
    def get_data(self, data_dict = None, s = (224,224), dataset = dai_image_csv_dataset, train_resize_transform = None, val_resize_transform = None, 
                 bs = 32, balance = False, super_res_crop = 256, super_res_upscale_factor = 1, sr_input_tfms = [], n_frames = 7,
                 tfms = [],bal_tfms = None,num_workers = 8, stats_percentage = 0.6,channels = 3, normalise = True, img_mean = None, img_std = None):
        
        self.image_size = s
        if not data_dict:
            data_dict = self.data_dict
        data_dfs, data_dir, single_label, seg, obj, sr= (data_dict['data_dfs'], data_dict['data_dir'],
                                                         data_dict['single_label'], data_dict['seg'],
                                                         data_dict['obj'], data_dict['sr'])
        if not single_label:
           balance = False                                                 
        if not bal_tfms:
            bal_tfms = { self.tr_name: [albu.HorizontalFlip()],
                         self.val_name: None,
                         self.test_name: None 
                       }
        else:
            bal_tfms = {self.tr_name: bal_tfms, self.val_name: None, self.test_name: None}

        # resize_transform = transforms.Resize(s,interpolation=Image.NEAREST)
        if train_resize_transform is None:
            train_resize_transform = albu.Resize(s[0],s[1],interpolation=2)
        if normalise:          
            if img_mean is None and self.img_mean is None: # and not sr:
                # temp_tfms = [resize_transform, transforms.ToTensor()]
                temp_tfms = [train_resize_transform, AT.ToTensor()]
                frac_data = data_dfs[self.tr_name].sample(frac = stats_percentage).reset_index(drop=True).copy()
                temp_dataset = dai_image_csv_dataset(data_dir = data_dir,data = frac_data,transforms_ = temp_tfms,channels = channels)
                self.img_mean,self.img_std = get_img_stats(temp_dataset,channels)
            elif self.img_mean is None:
                self.img_mean,self.img_std = img_mean,img_std
            normalise_transform = albu.Normalize(self.img_mean, self.img_std)
        else:
            normalise_transform = None
        # if obj:
        #     obj_transform = obj_utils.transform(size = s, mean = self.img_mean, std = self.img_std)
        #     dataset = dai_obj_dataset
        #     tfms = obj_transform.train_transform
        #     val_test_tfms = obj_transform.val_test_transform
        #     data_transforms = {
        #         self.tr_name: tfms,
        #         self.val_name: val_test_tfms,
        #         self.test_name: val_test_tfms
        #     }
        #     has_difficult = (len(data_dfs[self.tr_name].columns) == 4)
        #     image_datasets = {x: dataset(data_dir = data_dir,data = data_dfs[x],tfms = data_transforms[x],
        #                         has_difficult = has_difficult)
        #                       for x in [self.tr_name, self.val_name, self.test_name]}
        #     dataloaders = {x: DataLoader(image_datasets[x], batch_size=bs,collate_fn=image_datasets[x].collate_fn,
        #                                                 shuffle=True, num_workers=num_workers)
        #                 for x in [self.tr_name, self.val_name, self.test_name]}
        if sr:
            super_res_crop = super_res_crop - (super_res_crop % super_res_upscale_factor)
            super_res_transforms = {
                'pre_transforms':[albu.CenterCrop(super_res_crop,super_res_crop)]+tfms,
                'pre_input_transforms':sr_input_tfms,
                'downscale':
                            # [albu.OneOf([
                            #             albu.Resize((super_res_crop // super_res_upscale_factor),
                            #                         (super_res_crop // super_res_upscale_factor),
                            #                         interpolation = 0),
                            #             albu.Resize((super_res_crop // super_res_upscale_factor),
                            #                         (super_res_crop // super_res_upscale_factor),
                            #                         interpolation = 1),
                            #             albu.Resize((super_res_crop // super_res_upscale_factor),
                            #                         (super_res_crop // super_res_upscale_factor),
                            #                         interpolation = 2),
                            #             ],p=1)
                            # ],
                            [
                                albu.Resize((super_res_crop // super_res_upscale_factor),
                                            (super_res_crop // super_res_upscale_factor),
                                            interpolation = 2)
                            ],
                'input':[
                        # albu.CenterCrop(super_res_crop,super_res_crop),
                        albu.Resize((super_res_crop // super_res_upscale_factor),
                                    (super_res_crop // super_res_upscale_factor),
                                    interpolation = 2),
                        normalise_transform,
                        AT.ToTensor()
                ],
                'target':[
                        # albu.CenterCrop(super_res_crop,super_res_crop),
                        # albu.Normalize(self.img_mean,self.img_std),
                        normalise_transform,
                        AT.ToTensor()
                ],
                'resized_target':[
                    # albu.CenterCrop(super_res_crop,super_res_crop),
                    albu.Resize((super_res_crop // super_res_upscale_factor),
                                (super_res_crop // super_res_upscale_factor),
                                interpolation = 2),
                    albu.Resize(super_res_crop,super_res_crop,interpolation = 2),
                    # albu.OneOf([
                    #             albu.Resize(super_res_crop,super_res_crop,interpolation = 0),
                    #             albu.Resize(super_res_crop,super_res_crop,interpolation = 1),
                    #             albu.Resize(super_res_crop,super_res_crop,interpolation = 2),
                    #             ],p=1),
                    AT.ToTensor()
                ]
            }
            image_datasets = {x: dataset(data_dir=data_dir, data=data_dfs[x], transforms_=super_res_transforms, n_frames=n_frames)
                             for x in [self.tr_name, self.val_name, self.test_name]}

        else:   
            if len(tfms) == 0:
                # if normalise:
                #     tfms = [
                #         resize_transform,
                #         transforms.ToTensor(),
                #         transforms.Normalize(normalise_array,normalise_array)
                #     ]
                # else:
                #     tfms = [
                #         resize_transform,
                #         transforms.ToTensor()
                #     ]    
                tfms = [
                        train_resize_transform,
                        # transforms.ToTensor(),
                        # transforms.Normalize(self.img_mean,self.img_std)
                        normalise_transform,
                        AT.ToTensor()
                    ]
            else:
                tfms = [
                    train_resize_transform,
                    *tfms,
                    # transforms.ToTensor(),
                    # transforms.Normalize(self.img_mean,self.img_std)
                    normalise_transform,
                    AT.ToTensor()
                ]
                # tfms_temp[1:1] = tfms
                # tfms = tfms_temp
                print('Transforms: ',)
                print(tfms)
                print()
            if val_resize_transform is None:
                val_resize_transform = albu.Resize(s[0],s[1],interpolation=2)
            val_test_tfms = [
                val_resize_transform,
                # transforms.ToTensor(),
                # transforms.Normalize(self.img_mean,self.img_std)
                # normalise_transform,
                AT.ToTensor()
            ]
            data_transforms = {
                self.tr_name: tfms,
                self.val_name: val_test_tfms,
                self.test_name: val_test_tfms
            }

            # if balance:
            #     image_datasets = {x: dataset(data_dir = data_dir,data = data_dfs[x],
            #                                 transforms_ = data_transforms[x],minorities = minorities,diffs = class_diffs,
            #                                 bal_tfms = bal_tfms[x],channels = channels,seg = seg)
            #                 for x in [self.tr_name, self.val_name, self.test_name]}    
            if self.multi_head:
                dataset = dai_image_csv_dataset_multi_head
                image_datasets = {x: dataset(data_dir = data_dir,data = data_dfs[x],
                                            transforms_ = data_transforms[x],channels = channels)
                            for x in [self.tr_name, self.val_name, self.test_name]}
            else:
                image_datasets = {x: dataset(data_dir=data_dir, data=data_dfs[x], transforms_=data_transforms[x],
                                             input_transforms=data_transforms[x], target_transforms=data_transforms[x],
                                             channels=channels, seg=seg)
                            for x in [self.tr_name, self.val_name, self.test_name]}
        
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=bs, shuffle=True, num_workers=num_workers)
                       for x in [self.tr_name, self.val_name, self.test_name]}
        dataset_sizes = {x: len(image_datasets[x]) for x in [self.tr_name, self.val_name, self.test_name]}
        self.image_datasets,self.dataloaders,self.dataset_sizes = (image_datasets,dataloaders,dataset_sizes)
        
        return image_datasets,dataloaders,dataset_sizes