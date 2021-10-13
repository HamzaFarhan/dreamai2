from .utils import *
from .dai_imports import *

class DaiDataset(Dataset):
    
    def __init__(self, data, data_dir='', tfms=None, ss_tfms=None, channels=3,
                 meta_idx=None, do_tta=False, tta=None, num_tta=3, img_idx=0, label_idx=1,
                   **kwargs):
        super(DaiDataset, self).__init__()
        if is_df(data) and len(data.columns) == 1:
            data['extra_col'] = 'extra'
        store_attr(self,'data,tfms,ss_tfms,meta_idx,do_tta,tta,num_tta,channels,img_idx,label_idx')
        self.tfms_list = []
        self.data_dir = str(data_dir)
        if self.tfms is not None:
            self.tfms_list.append(self.tfms)
        if self.ss_tfms is not None:
            self.tfms_list.append(self.ss_tfms)
        for k in kwargs:
            setattr(self, k, kwargs[k])
        
    def __len__(self):
        return len(self.data)
    
    def get_name(self, index):
        try:
            img_name = self.data.iloc[index, self.img_idx]
        except:
            img_name = self.data[index, self.img_idx]
        return [img_name]

    def get_img_path(self, index):
        return [os.path.join(self.data_dir, x) for x in self.get_name(index)]

    def get_img(self, index):
        img_path = self.get_img_path(index)
        # print(f'lala {img_path}')
        try:
            if self.channels == 3:
                img = [rgb_read(x) for x in img_path]
            else:    
                img = [c1_read(x) for x in img_path]
            return img
        except:
            print(img_path)
    
    def get_y(self, index, str_to_index=True):
        try:
            y = self.data.iloc[index, self.label_idx]
        except:
            y = self.data[index, self.label_idx]
        # print(f'ooolala {y}')
        y2 = self.get_show_label(y, index)                 
        if str_to_index:
            if is_str(y) and hasattr(self, 'class_names'):
                y = self.class_names.index(y)
                # y = tensor(self.class_names.index(y))
        return y, y2

    def get_tta(self, **kwargs):
        img,y,name = kwargs['img'], kwargs['y'], kwargs['name']
        if self.tta is None:
            self.tta = [self.tfms] * self.num_tta
            # if self.tta is not None:
        ret_tta = [{'x':apply_tfms(img.copy(),t), 'label':y, 'name':name} for t in self.tta]
        return ret_tta

    def get_x(self, to_tensor=True, **kwargs):
        img = kwargs['img']
        if self.tfms is not None:
            x = [apply_tfms(i.copy(), self.tfms) for i in img]
            if self.channels == 1:
                x = [i.unsqueeze(0) for i in x]
            if not to_tensor:
                x = [tensor_to_img(i) for i in x]
        else:
            x = img
        return x

    def get_meta(self, index):
        if not list_or_tuple(self.meta_idx):
            meta1,meta2 = self.meta_idx, self.meta_idx+1
        else:
            meta1,meta2 = self.meta_idx
        try:
            ret_meta = torch.cat([torch.tensor(m).float() for m in self.data.iloc[index, meta1:meta2]]).float()
        except:
            ret_meta = torch.cat([torch.tensor([m]).float() for m in self.data.iloc[index, meta1:meta2]]).float()
        return ret_meta

    def get_ret(self, **kwargs):
        l = locals_to_params(locals())
        remove_key(l, lambda x: x not in ['x', 'y', 'y2', 'name'])
        change_key_name(l, 'y', 'label')
        change_key_name(l, 'y2', 'show_label')
        return l

    def __getitem__(self, index, to_tensor=True, str_to_index=True):
        # name = self.get_name(index=index)
        name = self.get_img_path(index=index)
        img = self.get_img(index=index)
        y,y2 = self.get_y(index=index, str_to_index=str_to_index)
        if self.do_tta:
            return self.get_tta(**locals_to_params(locals()))
        x = self.get_x(**locals_to_params(locals()))
        if is_list(x) and len(x) == 0:
            x = x[0]
        ret = self.get_ret(**locals_to_params(locals()))

        # if self.ss_tfms is not None:
            # ret['x2'], ret['ss_img'] = self.get_ss(to_tensor=to_tensor)

        if self.meta_idx is not None:
            ret['meta'] = self.get_meta(index=index)
            
        return ret

    def denorm_data(self, data):
        if self.tfms is not None:
            to_denorm = ['x']
            if self.ss_tfms:
                to_denorm.append('x2')
                to_denorm.append('ss_img')
            norm_t = get_norm(self.tfms)
            if norm_t:
                mean = norm_t.mean
                std = norm_t.std
                for k in to_denorm:
                    data[k] = denorm_img(data[k], mean, std)

    def show_data(self, data):
        x,name,label = data['x'], data['name'], data['show_label']
        print(f'Name:{name}')
        if self.tfms is None:
            aug = ''
        else: aug = ' Augmented'
        plt_show(x, title=f'Normal{aug}: {label}')
        if self.ss_tfms is not None:
            img2,x2 = data['ss_img'], data['x2']
            plt_show(img2, title=f'SS Image: {label}')
            plt_show(x2, title=f'SS Augmented: {label}')

    def get_show_label(self, y, index):
        if not is_str(y) and (is_iterable(y) or is_tensor(y)):
            label = self.data.iloc[index, self.label_idx+1]
        else:
            label = y
        return label

    def get_at_index(self, index, denorm=True, show=True, to_tensor=False):

        data = self.__getitem__(index, to_tensor=to_tensor, str_to_index=False)
        if denorm:
            self.denorm_data(data=data)
        if is_list(data['x']):
            data['x'] = np.concatenate(data['x'], axis=1)
        ret = data
        if show:
            self.show_data(data=data)
        return ret

def init_vis(dset, data_dir='', img_idx=0, label_idx=1):
    original_set = DaiDataset(data=dset.data, data_dir=data_dir)
    b1 = original_set.get_at_index(0)
    print(f"Shape: {b1['x'].shape}")
    b2 = dset.get_at_index(0)
    print(f"Shape: {b2['x'].shape}")
    return b1,b2

class MatchingDatasetOld(Dataset):
    
    def __init__(self, data, data_dir='', tfms=None, channels=3,
                 meta_idx=None, do_tta=False, tta=None, num_tta=3,  **kwargs):
        super(MatchingDatasetOld, self).__init__()
        self.tfms_list = []
        self.data_dir = str(data_dir)
        self.data = data
        self.tfms = tfms
        self.ss_tfms = ss_tfms
        self.meta_idx = meta_idx
        self.do_tta = do_tta
        self.tta = tta
        self.num_tta = num_tta
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
        # img_path = str(img_path)
        # if not Path(img_path).exists():
        #print(img_path)
        imgs = []
        for img_path in img_path.split():
            if self.channels == 3:
                img = rgb_read(img_path)
            else:    
                img = c1_read(img_path)
            imgs.append(img)

        try:
            y = self.data.iloc[index, 1]
        except:
            y = self.data[index, 1]                    

        if is_str(y) and hasattr(self, 'class_names'):
            y = self.class_names.index(y)

        if self.do_tta:
            if self.tta is None:
                self.tta = [self.tfms] * self.num_tta
            ret_tta = [{'x':apply_tfms(imgs[0].copy(),t), 'x2':apply_tfms(imgs[1].copy(),t),
                        'label':y, 'path':self.data.iloc[index, 0]} for t in self.tta]
            return ret_tta

        x = []
        for img in imgs:
            if self.tfms is not None:
                x_ = apply_tfms(img.copy(), self.tfms)
                if self.channels == 1:
                    x_ = x_.unsqueeze(0)
            else:
                x_ = img
            x.append(x_)

        x,x2 = x[0], x[1]
        # if len(x) == 1:
            # x = x[0]

        ret = {'x':x, 'x2':x2, 'label':y, 'path':self.data.iloc[index, 0]}

        # for i in range(1,len(x)):
            # ret[f'x{i+1}'] = x[i]

        if self.meta_idx is not None:
            if not list_or_tuple(self.meta_idx):
                meta1,meta2 = self.meta_idx, self.meta_idx+1
            else:
                meta1,meta2 = self.meta_idx
            try:
                ret['meta'] = torch.cat([torch.tensor(m).float() for m in self.data.iloc[index, meta1:meta2]]).float()
            except:
                ret['meta'] = torch.cat([torch.tensor([m]).float() for m in self.data.iloc[index, meta1:meta2]]).float()

        return ret
        # return {'x':x, 'label':y, 'ss_img':img2, 'x2':x2, 'path':self.data.iloc[index, 0]}
        # return x, y, img2, x2, self.data.iloc[index, 0]

    def get_at_index(self, index, denorm=True, show=True):

        got_item = self.__getitem__(index)
        y = self.data.iloc[index, 1]
        if not is_str(y) and is_iterable(y):
            label = self.data.iloc[index, 2]
        else:
            label = y
        gtx = [got_item['x'], got_item['x2']]
        for i,x in enumerate(gtx):
            if self.tfms is not None:
                # x = apply_tfms(img.copy(), self.tfms)
                # if self.channels == 1:
                    # x = x.unsqueeze(0)
                x = tensor_to_img(x)
                if denorm:
                    norm_t = get_norm(self.tfms)
                    if norm_t:
                        mean = norm_t.mean
                        std = norm_t.std
                        x = denorm_img(x, mean, std)
            else:
                x = img
            gtx[i] = x

        x,x2 = gtx[0], gtx[1]
        p = self.data.iloc[index, 0]

        ret = {'x':x, 'x2':x2, 'label':y, 'path':p}
                
        if show:
            print(f'path:{p}')
            if self.tfms is None:
                aug = ''
            else: aug = ' Augmented'
            for i,x_ in enumerate(gtx):
                plt_show(x_, title=f'Normal Img{i}{aug}: {label}')

        if self.meta_idx is not None:
            if not list_or_tuple(self.meta_idx):
                meta1,meta2 = self.meta_idx, self.meta_idx+1
            else:
                meta1,meta2 = self.meta_idx
            try:
                ret['meta'] = torch.cat([torch.tensor(m) for m in self.data.iloc[index, meta1:meta2]])
            except:
                ret['meta'] = torch.cat([torch.tensor([m]) for m in self.data.iloc[index, meta1:meta2]])
        return ret
        # return {'x':x, 'label':y, 'ss_img':img2, 'x2':x2, 'path':p}

class SimilarityDataset(Dataset):
    
    def __init__(self, data, data_dir='', tfms=None, tfms2=None, channels=3, shuffle_data2=True, same_img=False, **kwargs):
        super(SimilarityDataset, self).__init__()
        self.tfms_list = []
        self.data_dir = str(data_dir)
        self.data = data
        self.shuffle_data2 = shuffle_data2
        self.same_img = same_img
        if shuffle_data2:
            self.data2 = data.copy()
        else:
            self.data2 = data.copy().sample(frac=1., random_state=2).reset_index(drop=True)
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
        # img_path = str(img_path)
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
        if self.shuffle_data2:
            index2 = random.choice(range(len(self.data2)))
        else:
            index2 = index
        if self.same_img:
            self.data2 = self.data.copy()
            index2 = index
        try:
            y2 = self.data2.iloc[index2, 1]
        except:
            y2 = self.data2[index2, 1]
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

        # while y2 != y:
        #     index2 = random.choice(range(len(self.data)))
        #     try:
        #         y2 = self.data.iloc[index2, 1]
        #     except:
        #         y2 = self.data[index2, 1]
        #     if is_str(y2) and hasattr(self, 'class_names'):
        #         y2 = self.class_names.index(y2)
        try:
            img_path2 = os.path.join(self.data_dir, self.data2.iloc[index2, 0])
        except:
            img_path2 = os.path.join(self.data_dir, self.data2[index2, 0])
        # img_path2 = str(img_path2)
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
               'path2':self.data2.iloc[index2, 0], 'same':same}

        return ret

    def get_at_index(self, index, denorm=True, show=True):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        # img_path = str(img_path)
        if self.channels == 3:
            img = rgb_read(img_path)
        else:    
            img = c1_read(img_path)
        
        y = self.data.iloc[index, 1]
        if not is_str(y) and is_iterable(y):
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

        if self.shuffle_data2:
            index2 = random.choice(range(len(self.data2)))
        else:
            index2 = index
        if self.same_img:
            self.data2 = self.data.copy()
            index2 = index
        try:
            y2 = self.data2.iloc[index2, 1]
        except:
            y2 = self.data2[index2, 1]

        # while y2 != y:
        #     index2 = random.choice(range(len(self.data)))
        #     try:
        #         y2 = self.data.iloc[index2, 1]
        #     except:
        #         y2 = self.data[index2, 1]
        
        if not is_str(y2) and is_iterable(y2):
            label2 = self.data2.iloc[index2, 2]
        else:
            label2 = y2        

        try:
            img_path2 = os.path.join(self.data_dir, self.data2.iloc[index2, 0])
        except:
            img_path2 = os.path.join(self.data_dir, self.data2[index2, 0])
        # img_path2 = str(img_path2)
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

        p2 = self.data2.iloc[index2, 0]

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

class MatchingDataset(Dataset):
    
    def __init__(self, data, data_dir='', tfms=None, tfms2=None, channels=3, shuffle_data2=True, same_img=False, **kwargs):
        super(MatchingDataset, self).__init__()
        self.tfms_list = []
        self.data_dir = str(data_dir)
        self.data = data
        self.shuffle_data2 = shuffle_data2
        self.same_img = same_img
        if shuffle_data2:
            self.data2 = data.copy()
        else:
            self.data2 = data.copy().sample(frac=1., random_state=2).reset_index(drop=True)
        self.tfms = tfms
        self.tfms2 = tfms2
        if self.tfms is not None:
            self.tfms_list.append(self.tfms)
        if self.tfms2 is not None:
            self.tfms_list.append(self.tfms2)
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
            if self.channels == 1:
                x = x.unsqueeze(0)
        else:
            x = img
        if is_str(y) and hasattr(self, 'class_names'):
            y = self.class_names.index(y)
        # ret = {'x':x, 'label':y, 'path':self.data.iloc[index, 0]}
        if self.shuffle_data2:
            index2 = random.choice(range(len(self.data2)))
        else:
            index2 = index
        if self.same_img:
            self.data2 = self.data.copy()
            index2 = index
        try:
            y2 = self.data2.iloc[index2, 1]
        except:
            y2 = self.data2[index2, 1]
        if is_str(y2) and hasattr(self, 'class_names'):
            y2 = self.class_names.index(y2)
        try:
            if y2 == y:
                same = 1
            else:
                same = 0
        except:
            if (y2 == y).all():
                same = 1
            else:
                same = 0
        try:
            img_path2 = os.path.join(self.data_dir, self.data2.iloc[index2, 0])
        except:
            img_path2 = os.path.join(self.data_dir, self.data2[index2, 0])
        # img_path2 = str(img_path2)
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
        
        ret = {'x':x, 'label':same, 'path':self.data.iloc[index, 0], 'x2':x2,
               'path2':self.data2.iloc[index2, 0], 'same':same}

        return ret

    def get_at_index(self, index, denorm=True, show=True):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        # img_path = str(img_path)
        if self.channels == 3:
            img = rgb_read(img_path)
        else:    
            img = c1_read(img_path)
        
        y = self.data.iloc[index, 1]
        if not is_str(y) and is_iterable(y):
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

        if self.shuffle_data2:
            index2 = random.choice(range(len(self.data2)))
        else:
            index2 = index
        if self.same_img:
            self.data2 = self.data.copy()
            index2 = index
        try:
            y2 = self.data2.iloc[index2, 1]
        except:
            y2 = self.data2[index2, 1]

        if not is_str(y2) and is_iterable(y2):
            label2 = self.data2.iloc[index2, 2]
        else:
            label2 = y2        

        try:
            img_path2 = os.path.join(self.data_dir, self.data2.iloc[index2, 0])
        except:
            img_path2 = os.path.join(self.data_dir, self.data2[index2, 0])
        # img_path2 = str(img_path2)
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

        p2 = self.data2.iloc[index2, 0]

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
    def __init__(self, data, data_dir='', tfms=None, channels=3, meta_idx=None,
                 do_tta=False, tta=None, num_tta=3, **kwargs):
        super(PredDataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.tfms = tfms
        self.channels = 3
        self.meta_idx = meta_idx
        self.do_tta = do_tta
        self.tta = tta
        self.num_tta = num_tta
        for k in kwargs:
            setattr(self, k, kwargs[k])
    
    def __getitem__(self, index):

        try:
            img_path = os.path.join(self.data_dir, self.data.iloc[index,0])
            # img_path = str(img_path)
            if self.channels == 3:
                img = rgb_read(img_path)
            else:    
                img = c1_read(img_path)
            name = img_path
        except:
            img = self.data.iloc[index,0]
            name = 'img_0'

        if self.do_tta:
            if self.tta is None:
                self.tta = [self.tfms] * self.num_tta
            # if self.tta is not None:
            ret_tta = [{'x':apply_tfms(img.copy(),t)} for t in self.tta]
            return ret_tta

        if self.tfms is not None:
            x = self.tfms(image=img)['image']
            if self.channels == 1:
                x = x.unsqueeze(0)
        else:
            x = img
        ret = {'name':name}
        ret['x'] = x
        if self.meta_idx is not None:
            if not list_or_tuple(self.meta_idx):
                meta1,meta2 = self.meta_idx, self.meta_idx+1
            else:
                meta1,meta2 = self.meta_idx
            try:
                ret['meta'] = torch.cat([torch.tensor(m).float() for m in self.data.iloc[index, meta1:meta2]]).float()
            except:
                ret['meta'] = torch.cat([torch.tensor([m]).float() for m in self.data.iloc[index, meta1:meta2]]).float()
        return ret

    def __len__(self): return len(self.data)

class Similarity():
    def __init__(self, model, dset=PredDataset, tfms=instant_tfms(224,224), ems=[]):
        self.model = model
        self.dset = dset
        self.tfms = tfms
        self.imgs = []
        self.nms = None
        self.ems = ems
        
        if len(ems) > 0:
            data = np.array(self.ems)
            self.nms = nmslib.init(method='hnsw', space='cosinesimil')
            self.nms.addDataPointBatch(data)
            self.nms.createIndex({'post': 2}, print_progress=True) 
    
    def get_embeddings(self, x, device=None):
        
        if device is None:
            device = default_device()
        
        if path_or_str(x):
            x = rgb_read(str(x))
        if is_array(x):
            if x.ndim == 3:
                x = [x]
            else:
                x = list(x)
            x = pd.DataFrame({'x': x}, columns=['x'])
        elif is_tensor(x):
            x = tensor_to_img(x)
            if is_array(x):
                x = [x]
            x = pd.DataFrame({'x': x}, columns=['x'])
        
        data = self.dset(x, tfms=self.tfms)
        #loader = DataLoader(data, batch_size=bs, num_workers=6, shuffle=False)
        self.model.to(device).eval()
        embeddings = []
        for idx,data_batch in enumerate(data):
            batchify_dict(data_batch)
            pred_out = self.model.get_embeddings(data_batch, device=device).detach().cpu()
            embeddings.append(pred_out)
        return embeddings
    
    def insert(self, image, images_path='opt/images', embedding_path='opt/embeddings', embedding_name='ems.parquet',
               nms_path='opt/nms', num_neighbours=3, show=False, get_neighbours=True, device=None):
        
        if device is None:
            device = default_device()
            
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(embedding_path, exist_ok=True)
        os.makedirs(nms_path, exist_ok=True)
        
        image = Path(image)
        images_path = Path(images_path)
        embedding_path = Path(embedding_path)
        nms_path = Path(nms_path)
        
        self.imgs.append(image)
        name = image.name
        img_em = self.get_embeddings(image, device=device)
        em = torch.cat(img_em).squeeze(0).numpy()
        
        if self.nms is not None and get_neighbours:
            num_images = len(self.ems)
            if num_neighbours > num_images:
                print(f'Only {num_images} image(s) available. So number of neighbours = {num_images}.')
                num_neighbours = num_images
            ids, distances = self.nms.knnQuery(em, k=num_neighbours)
            # print(ids)
            if show:
                plt_show(rgb_read(image), title='Query Image')
                sim_imgs = np.array(self.imgs)[ids]
                plot_in_row([albu_resize(rgb_read(i), 224,224) for i in sim_imgs],
                            rows=1, columns=len(sim_imgs),
                            titles=[f'sim_{i+1}\ndistance: {d}' for i,d in enumerate(distances)])
                
        self.ems.append(em)
        data = np.array(self.ems)
        self.nms = nmslib.init(method='hnsw', space='cosinesimil')
        self.nms.addDataPointBatch(data)
        self.nms.createIndex({'post': 2}, print_progress=True)

        np_to_parquet(self.ems, embedding_path/embedding_name)

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
        # img_path = str(img_path)
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
                       pin_memory=False, num_workers=4, force_one_hot=False, meta_idx=None,
                       class_names=None, split=True, val_size=0.2, test_size=0.15,
                       tta=None, num_tta=3, multi_delim=' ', **kwargs):

    if tta is not None and not list_or_tuple(tta):
        tta = [tta]*num_tta
    if not is_iterable(tfms):
        tfms = [tfms]
    def df_one_hot(df):
        df = df.copy()
        # labels = list(df.iloc[:,1].apply(lambda x: str(x).split()))
        labels = list_map(df.iloc[:,1], lambda x:str(x).split(multi_delim))
        one_hot_labels = dai_one_hot(labels, class_names)
        df['one_hot'] = list(one_hot_labels)
        cols = df.columns.to_list()
        df = df[[cols[0], cols[-1], *cols[1:-1]]]
        return df
    if len(df.columns) == 1:
        df['extra_col'] = 'extra'
    labels = list(df.iloc[:,1].apply(lambda x: str(x).split(multi_delim)))
    # labels = list_map(df.iloc[:,1], lambda x:str(x).split())
    # is_multi = np.array(pd.Series(labels).apply(lambda x:len(x)>1)).any()
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
        if force_one_hot:
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
    dsets = [dset(data_dir=data_dir, data=df, tfms=tfms_, tta=tta, num_tta=num_tta, **kwargs,
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

def get_regression_dls(df, val_df=None, test_df=None, data_dir='', dset=DaiDataset,
                       tfms=instant_tfms(224, 224), bs=64, shuffle=True,
                       pin_memory=False, num_workers=4, meta_idx=None,
                       split=True, val_size=0.2, test_size=0.15, crit=None, **kwargs):

    if not is_iterable(tfms):
        tfms = [tfms]
    if len(df.columns) == 1:
        df['extra_col'] = 'extra'
    labels = list(df.iloc[:,1])
    stratify_idx = 1
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
    dsets = [dset(data_dir=data_dir, data=df, tfms=tfms_, meta_idx=meta_idx, **kwargs) for df,tfms_ in zip(dfs, transforms_)]
    dls = get_dls(dsets=[dsets[0]], bs=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if split:
        dls += get_dls(dsets=dsets[1:], bs=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dls = DataLoaders(*dls)
    dls.data_type = 'regression'
    if crit is None:
        dls.suggested_crit = nn.MSELoss()
    else:
        dls.suggested_crit = crit
    dls.suggested_metric = 'loss'
    return dls

def get_matching_dls_old(df, val_df=None, test_df=None, data_dir='', dset=MatchingDataset,
                       tfms=instant_tfms(224, 224), bs=64, shuffle=True,
                       pin_memory=False, num_workers=4, meta_idx=None,
                       class_names=None, split=True, val_size=0.2, test_size=0.15,
                       tta=None, num_tta=3, **kwargs):

    if tta is not None and not list_or_tuple(tta):
        tta = [tta]*num_tta

    labels = list_map(df.iloc[:,1], lambda x:str(x).split())
    if class_names is None:
        class_names = np.unique(flatten_list(labels))
    class_names = list_map(class_names, str)
    stratify_idx = 1
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

    dsets = [dset(data_dir=data_dir, data=df, tfms=tfms_, tta=tta, num_tta=num_tta, **kwargs,
                  class_names=class_names, meta_idx=meta_idx) for df,tfms_ in zip(dfs, transforms_)]
    dls = get_dls(dsets=[dsets[0]], bs=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if split:
        dls += get_dls(dsets=dsets[1:], bs=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dls = DataLoaders(*dls)
    dls.class_names = class_names
    dls.num_classes = len(class_names)
    dls.class_weights = [get_class_weights(df) for df in dfs]
    dls.data_type = 'matching'
    dls.suggested_crit = nn.CrossEntropyLoss()
    dls.suggested_metric = 'accuracy'

    return dls

def get_similarity_dls(df, val_df=None, test_df=None, data_dir='', dset=SimilarityDataset,
                       tfms=instant_tfms(224, 224), tfms2=None, bs=64, shuffle=True, same_img=False,
                       pin_memory=False, num_workers=4, force_one_hot=False, shuffle_data2=True,
                       class_names=None, split=True, val_size=0.2, test_size=0.15, **kwargs):

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
    if tfms2 is None:
        transforms_2 = [None] * len(transforms_)
    else:
        if len(transforms_) > 1:
            transforms_2 = [tfms2] + [jigsaw_tfms(t) for t in transforms_[1:]]
        elif len(transforms_) == 1:
            transforms_2 = [tfms2]
    dsets = [dset(data_dir=data_dir, data=df, tfms=tfms_, tfms2=tfms_2, shuffle_data2=shuffle_data2, same_img=same_img, **kwargs,
                  class_names=class_names) for df,tfms_,tfms_2 in zip(dfs, transforms_, transforms_2)]
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

def get_matching_dls(df, val_df=None, test_df=None, data_dir='', dset=MatchingDataset,
                       tfms=instant_tfms(224, 224), tfms2=None, bs=64, shuffle=True, same_img=False,
                       pin_memory=False, num_workers=4, force_one_hot=False, shuffle_data2=True,
                       class_names=None, split=True, val_size=0.2, test_size=0.15, **kwargs):
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
    if tfms2 is None:
        transforms_2 = [None] * len(transforms_)
    else:
        if len(transforms_) > 1:
            transforms_2 = [tfms2] + [jigsaw_tfms(t) for t in transforms_[1:]]
        elif len(transforms_) == 1:
            transforms_2 = [tfms2]
    dsets = [dset(data_dir=data_dir, data=df, tfms=tfms_, tfms2=tfms_2, shuffle_data2=shuffle_data2, same_img=same_img, **kwargs,
                  class_names=class_names) for df,tfms_,tfms_2 in zip(dfs, transforms_, transforms_2)]
    dls = get_dls(dsets=[dsets[0]], bs=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if split:
        dls += get_dls(dsets=dsets[1:], bs=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dls = DataLoaders(*dls)
    class_names = ['same', 'not_same']
    dls.class_names = class_names
    dls.num_classes = len(class_names)
    dls.is_multi = is_multi
    dls.data_type = 'matching'
    dls.is_one_hot = is_multi + force_one_hot
    dls.suggested_crit = nn.CrossEntropyLoss()
    dls.suggested_metric = 'accuracy'

    return dls

class DataLoaders():
    def __init__(self, train=None, valid=None, test=None, remove_norm=False):
        store_attr(self, 'train,valid,test')
        if self.train:
            self.train_ds = self.train.dataset
            self.train_dl = partial(DataLoader, dataset=self.train_ds, batch_size=self.train.batch_size,
                                    shuffle=is_shuffle(self.train), num_workers=self.train.num_workers,
                                    pin_memory=self.train.pin_memory)

            norm_t, norm_id = get_norm_id(self.train_ds.tfms)
            if norm_id is not None:
                self.normalize = partial(apply_tfms, tfms=norm_t)
                self.denorm = partial(denorm_img, mean=norm_t.mean, std=norm_t.std)
                self.img_mean = norm_t.mean
                self.img_std = norm_t.std
                if remove_norm:
                    del_norm(self.train_ds.tfms, norm_id)
            else:
                self.normalize = noop
                self.denorm = noop
                self.img_mean, self.img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        if self.valid:
            self.valid_ds = self.valid.dataset
            self.valid_dl = partial(DataLoader, dataset=self.valid_ds, batch_size=self.valid.batch_size,
                                    shuffle=is_shuffle(self.valid), num_workers=self.valid.num_workers,
                                    pin_memory=self.valid.pin_memory)
            if remove_norm:
                norm_t, norm_id = get_norm_id(self.valid_ds.tfms)
                if norm_id is not None:
                    del_norm(self.valid_ds.tfms, norm_id)

        if self.test:
            self.test_ds = self.test.dataset
            self.test_dl = partial(DataLoader, dataset=self.test_ds, batch_size=self.test.batch_size,
                                   shuffle=is_shuffle(self.test), num_workers=self.test.num_workers,
                                   pin_memory=self.test.pin_memory)
            if remove_norm:
                norm_t, norm_id = get_norm_id(self.test_ds.tfms)
                if norm_id is not None:
                    del_norm(self.test_ds.tfms, norm_id)

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

    def __len__(self):
        return sum([(self.train!=None) + (self.valid!=None) + (self.test!=None)])

def get_dls(dsets, bs=32, shuffle=True, num_workers=4, pin_memory=False, collate_fn=None, ddp=None):
    dls = []
    for dset in dsets:
        if ddp is not None:
            sampler = ddp(dset)
            dls.append(DataLoader(dset, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler))
        else:
            dls.append(DataLoader(dset, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory))
    return dls

# def get_dls(dsets, bs=32, shuffle=True, num_workers=4, pin_memory=False, collate_fn=None):
#     dls = [DataLoader(dset, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory) for dset in dsets]
#     return dls

def get_class_weights(df):
    df = df.copy()
    try:
        counts = sum(df.iloc[:,1])
        total = sum(counts)
        w = [(x/total) for x in counts]
    except:
        w = list(df.iloc[:,1].value_counts(normalize=True).sort_index())    
    return 1-torch.tensor(w)

def get_data_stats(df, data_dir='', image_size=224, stats_percentage=0.7, bs=32, device=None,
                   img_idx=0, label_idx=1):
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print('Calculating dataset mean and std. This may take a while.', end='')
    print('Calculating dataset mean and std. This may take a while.\n')
    frac_data = df.copy().sample(frac=stats_percentage).reset_index(drop=True).copy()
    tfms = instant_tfms(image_size, image_size)[1]
    dset = DaiDataset(frac_data, data_dir=data_dir, tfms=tfms, img_idx=img_idx, label_idx=label_idx)
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
        if is_list(data_batch['x']):
            images = data_batch['x'][0].to(device)
        else:
            images = data_batch['x'].to(device)
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.to(dtype=torch.float32).mean(2).sum(0)
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
        if is_list(data_batch['x']):
            images = data_batch['x'][0].to(device)
        else:
            images = data_batch['x'].to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(dl.dataset)*image_size*image_size))
    print('\nDone.')
    return mean.cpu(), std.cpu()

def folders_to_dfs(images_path='', train_name=None, valid_name=None, test_name=None,
                   full_path=True, shuffle=False, do_str=True,
                   label_fn=lambda x: x.parent.name, folders=None):
    
    train_df = None
    valid_df = None
    test_df = None

    if train_name is not None:    
        imgs = get_image_files(Path(images_path)/train_name, folders=folders)
        # labels = list_map(imgs, label_fn)    
        # if not full_path:
        #     imgs = [end_of_path(x) for x in imgs]
        # if do_str:
        #     imgs = list_map(imgs, str)
        # train_df = pd.DataFrame({'img':imgs, 'label': labels}, columns=['img', 'label'])
        # if shuffle:
        #     train_df = train_df.sample(frac=1., random_state=2).reset_index(drop=True)
    else:
        imgs = get_image_files(images_path, folders=folders)
    labels = list_map(imgs, label_fn)    
    if not full_path:
        imgs = [end_of_path(x) for x in imgs]
    if do_str:
        imgs = list_map(imgs, str)
    train_df = pd.DataFrame({'img':imgs, 'label': labels}, columns=['img', 'label'])
    if shuffle:
        train_df = train_df.sample(frac=1., random_state=2).reset_index(drop=True)
    if valid_name is not None:
        imgs = get_image_files(Path(images_path)/valid_name, folders=folders)
        labels = list_map(imgs, label_fn)
        if not full_path:
            imgs = [end_of_path(x) for x in imgs]
        if do_str:
            imgs = list_map(imgs, str)
        valid_df = pd.DataFrame({'img':imgs, 'label': labels}, columns=['img', 'label'])
        if shuffle:
            valid_df = valid_df.sample(frac=1., random_state=2).reset_index(drop=True)
    if test_name is not None:
        imgs = get_image_files(Path(images_path)/test_name, folders=folders)
        labels = list_map(imgs, label_fn)
        if not full_path:
            imgs = [end_of_path(x) for x in imgs]
        if do_str:
            imgs = list_map(imgs, str)
        test_df = pd.DataFrame({'img':imgs, 'label': labels}, columns=['img', 'label'])
        if shuffle:
            test_df = test_df.sample(frac=1., random_state=2).reset_index(drop=True)
    return train_df, valid_df, test_df



































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
            y2 = torch.tensor(literal_eval(y2))
        except:
            y2 = torch.tensor(y2)
        self.tfms = transforms.Compose(self.transforms_)    
        x = self.tfms(img)
        s = x.shape[1]
        if isinstance(s,tuple):
            s = s[0]
        row_scale = s/img.size[1]
        col_scale = s/img.size[0]
        y2 = ((rescale_landmarks(copy.deepcopy(y2),row_scale,col_scale).squeeze())-s)/s
        return (x,y1,y2)


def rescale_landmarks(landmarks,row_scale,col_scale):
    landmarks2 = copy.deepcopy(torch.tensor(landmarks).reshape((-1,2)))
    for lm in landmarks2:
        c,r = lm
        lm[0] = c*col_scale
        lm[1] = r*row_scale
        # lm[0] = c*row_scale
        # lm[1] = r*col_scale
    landmarks2 = landmarks2.reshape((1,-1))        
    return landmarks2