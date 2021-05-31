from .utils import *
# from .obj import *
from .plot_eval import *

def efficientnet_b0(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b0',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b2(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b2',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b4(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b4',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b5(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b5',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b6(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b6',
           num_classes=num_channels, in_channels=in_channels)
def efficientnet_b7(num_channels=10, in_channels=3, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b7',
           num_classes=num_channels, in_channels=in_channels)

models_meta = {resnet34: {'cut': -2, 'conv_channels': 512},
               resnet50: {'cut': -2, 'conv_channels': 2048},
               resnet101: {'cut': -2, 'conv_channels': 2048},
               resnext50_32x4d: {'cut': -2, 'conv_channels': 2048},
               resnext101_32x8d: {'cut': -2, 'conv_channels': 2048},
               densenet121: {'cut': -1, 'conv_channels': 1024},
               densenet169: {'cut': -1, 'conv_channels': 1664},
               efficientnet_b0: {'cut': -5, 'conv_channels': 1280},
               efficientnet_b2: {'cut': -5, 'conv_channels': 1408},
               efficientnet_b4: {'cut': -5, 'conv_channels': 1792},
               efficientnet_b5: {'cut': -5, 'conv_channels': 2048},
               efficientnet_b6: {'cut': -5, 'conv_channels': 2304},
               efficientnet_b7: {'cut': -5, 'conv_channels': 2560}}

DEFAULTS = {'models_meta': models_meta, 'metrics': ['loss', 'accuracy', 'multi_accuracy'],
            'imagenet_stats': imagenet_stats, 'image_extensions': image_extensions}

class BodyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if isinstance(self.model, EfficientNet):
            return self.model.extract_features(x)
        return self.model(x)

def create_body(arch, pretrained=True, cut=None, num_extra=3):
    model = arch(pretrained=pretrained)
    if isinstance(model, EfficientNet):
        body_model = BodyModel(model)
    else:
        if cut is None:
            ll = list(enumerate(model.children()))
            cut = next(i for i,o in reversed(ll) if has_pool_type(o))
        modules = list(model.children())[:cut]
        body_model = BodyModel(nn.Sequential(*modules))
    if num_extra > 0:
        channels = models_meta[arch]['conv_channels']
        extra_convs = [conv_block(channels, channels)]*num_extra
        extra_model = nn.Sequential(*extra_convs)
        body_model = nn.Sequential(body_model, extra_model)
    else:
        body_model = nn.Sequential(body_model)
    return body_model

class HeadModel(nn.Module):
    def __init__(self, pool, linear):
        super().__init__()
        store_attr(self, 'pool,linear')
    def forward(self, x, meta=None):
        if meta is None:
            return self.linear(self.pool(x))  
        return self.linear(torch.cat([self.pool(x), meta], dim=1))

class MultiHeadModel(nn.Module):
    def __init__(self, head_list):
        super().__init__()
        self.head_list = head_list
    def forward(self, x, meta=None):
        return [h(x, meta) for h in self.head_list]

def create_head(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True,
                bn_final=False, lin_first=False, y_range=None, actv=None,
                relu_fn=nn.ReLU(inplace=True)):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    ps = [ps]
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [relu_fn] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    pool_layers = nn.Sequential(*[pool, Flatten()])
    layers = []
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += LinBnDrop(ni, no, bn=True, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if actv is not None:
        layers.append(actv)
    layers = nn.Sequential(*layers)
    return HeadModel(pool=pool_layers, linear=layers)

def create_model(arch, num_classes, num_extra=3, meta_len=0, body_out_mult=1,
                 relu_fn=nn.ReLU(inplace=True), actv=None, pretrained=True,
                 only_body=False, state_dict=None, strict_load=True):
    meta = models_meta[arch]
    body = create_body(arch, pretrained=pretrained, cut=meta['cut'], num_extra=num_extra)
    if only_body:
        return body
    if is_iterable(num_classes):
        heads = []
        for nc in num_classes:
            heads.append(create_head(nf=((meta['conv_channels']*2)*body_out_mult)+meta_len, n_out=nc, relu_fn=relu_fn, actv=actv)) 
        head = MultiHeadModel(nn.ModuleList(heads))
    else:
        head = create_head(nf=((meta['conv_channels']*2)*body_out_mult)+meta_len, n_out=num_classes, relu_fn=relu_fn, actv=actv)
    net = nn.Sequential(body, head)
    load_state_dict(net, sd=state_dict, strict=strict_load)
    return net

# def model_splitter(model, extra_cut=3, only_body=False):
#     if not only_body:
#         if extra_cut != 0:
#             return params(model[0][:-extra_cut]), params(model[0][-extra_cut:]) + params(model[1])
#         return params(model[0]), params(model[1])
#     if extra_cut == 0:
#         extra_cut = len(params(model))//6
#     return params(model[:-extra_cut]), params(model[-extra_cut:])

def model_splitter(model, cut_percentage=0.2, only_body=False):
    
    if not is_sequential(model):
        p = params(model)
        cut = int(len(p)*(1-cut_percentage))
        ret1 = p[:cut]
        ret2 = p[cut:]
        return ret1,ret2

    elif len(model) > 2:
        p = params(model)
        cut = int(len(p)*(1-cut_percentage))
        ret1 = p[:cut]
        ret2 = p[cut:]
        return ret1,ret2
        
    if not only_body:
        ret1, ret2 = params(model[0]), params(model[1])            
        p = params(model[0][0])
        cut = int(len(p)*(1-cut_percentage))
        ret1 = p[:cut]
        ret2 = p[cut:] + params(model[1])
        if len(model[0]) > 1:
            # print('yeesdsssss')
            ret2 += params(model[0][1])
        return ret1, ret2

    if cut_percentage == 0.:
        print("Must pass a cut percentage in the case of 'only_body'. Setting it to 0.2.")
        cut_percentage = 0.2
    p = params(model[0])
    cut = int(len(p)*(1-cut_percentage))
    ret1 = p[:cut]
    ret2 = p[cut:]
    if len(model) > 1:
        ret2 += params(model[1])
    return ret1,ret2

class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, loss):
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1-pt)**self.gamma * loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def focal_loss_(loss, alpha=1, gamma=2, reduce=True):
    pt = torch.exp(-loss)
    F_loss = alpha * (1-pt)**gamma * loss
    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss

# def focal_loss(alpha=1, gamma=2, reduce=True):
    # return partial(focal_loss_, alpha=alpha, gamma=gamma, reduce=reduce)

class Printer(nn.Module):
    def forward(self,x):
        print(x.size())
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
               relu=True, bn=True, dropout=True, dropout_p=0.2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if relu: layers.append(nn.ReLU(True))
    if bn: layers.append(nn.BatchNorm2d(out_channels))
    if dropout: layers.append(nn.Dropout2d(dropout_p))
    return nn.Sequential(*layers) 

def cnn_input(o,k,s,p):
    return ((o*s)-s)+k-(2*p) 

def cnn_output(w,k,s,p):
    return np.floor(((w-k+(2*p))/s))+1

def cnn_stride(w,o,k,p):
    return np.floor((w-k+(2*p))/(o-1))

def cnn_padding(w,o,k,s):
    return np.floor((((o*s)-s)-w+k)/2 )

class Classifier():
    def __init__(self, class_names):
        self.class_names = class_names
        if is_list(class_names) and is_list(class_names[0]):
            self.class_correct = []
            self.class_totals = []
            for _ in range(len(class_names)):
                self.class_correct+=[defaultdict(int)]
                self.class_totals+=[defaultdict(int)]
        else:
            self.class_correct = defaultdict(int)
            self.class_totals = defaultdict(int)

    def update_accuracies(self, outputs, labels):

        def update_accuracies_(outputs, labels, class_correct, class_totals):
            _, preds = torch.max(torch.exp(outputs), 1)
            correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
            for i in range(labels.shape[0]):
                label = labels.data[i].item()
                try:
                    class_correct[label] += correct[i].item()
                except:
                    class_correct[label] += 0
                class_totals[label] += 1

        if is_list(outputs):
            for i,(o,l) in enumerate(zip(outputs, labels)):
                update_accuracies_(o, l, self.class_correct[i], self.class_totals[i])
        else:
            update_accuracies_(outputs, labels, self.class_correct, self.class_totals)

    def update_multi_accuracies(self, outputs, labels, thresh=0.5):

        def update_multi_accuracies_(outputs, labels, class_correct, class_totals):
            preds = torch.sigmoid(outputs) > thresh
            correct = (labels==1)*(preds==1)
            for i in range(labels.shape[0]):
                label = torch.nonzero(labels.data[i]).squeeze(1)
                for l in label:
                    c = correct[i][l].item()
                    l = l.item()
                    class_correct[l] += c
                    class_totals[l] += 1
                    # self.class_correct[l] += c
                    # self.class_totals[l] += 1

        if is_list(outputs):
            for i,(o,l) in enumerate(zip(outputs, labels)):
                update_multi_accuracies_(o, l, self.class_correct[i], self.class_totals[i])
        else:
            update_multi_accuracies_(outputs, labels, self.class_correct, self.class_totals)

    def get_final_accuracies(self):

        def get_final_accuracies_(class_correct, class_totals, class_names):
            # print(class_correct, class_totals, class_names)
            accuracy = (100*np.sum(list(class_correct.values()))/np.sum(list(class_totals.values())))
            try:
                class_accuracies = [(class_names[i],100.0*(class_correct[i]/class_totals[i])) 
                                    for i in class_names.keys() if class_totals[i] > 0]
            except:
                class_accuracies = [(class_names[i],100.0*(class_correct[i]/class_totals[i])) 
                                    for i in range(len(class_names)) if class_totals[i] > 0]
            return accuracy, class_accuracies

        if is_list(self.class_correct):
            accuracy = []
            class_accuracies = []
            for class_correct, class_totals, class_names in zip(self.class_correct, self.class_totals, self.class_names):
                a,ca = get_final_accuracies_(class_correct, class_totals, class_names)
                accuracy.append(a)
                class_accuracies.append(ca)
        else:
            accuracy, class_accuracies = get_final_accuracies_(self.class_correct, self.class_totals, self.class_names)
        return accuracy, class_accuracies

class ConfusionMatrix():
    def __init__(self, matrix, class_names):
        self.matrix = matrix
        self.class_names = class_names

    def show(self, figsize=[8,8]):
        plot_confusion_matrix(self.matrix, self.class_names, figsize=figsize)

class ClassificationReport():
    def __init__(self, report, accuracy, class_accuracies):
        self.str_report = report

        split = report.split('\n')
        scores = [x for x in split[0].split(' ') if len(x)>0]
        report = {}
        report['Overall Accuracy'] = accuracy
        l = list_map(split[2:-5], lambda x:x.split())
        class_names = [x[0] for x in l]
        vals = [x[1:] for x in l]
        for i,val in enumerate(vals):
            scores_dict = {}
            for j,s in enumerate(val):
                scores_dict[scores[j]] = float(s)
            scores_dict['accuracy'] = class_accuracies[i][1]
            report[class_names[i]] = scores_dict
        self.report = report

    def show(self):
        plot_classification_report(self.str_report)

class BasicModel(nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head
        self.model = nn.Sequential(body, head)
    def forward(self, x):
        return self.model(x)
    def split_params(self):
        return self.body.parameters(), self.head.parameters()

class DaiModel(nn.Module):
    def __init__(self, model, opt=None, crit=nn.BCEWithLogitsLoss(), pred_thresh=0.5,
                 device=None, checkpoint=None, load_opt=False, load_crit=False, load_misc=False, **kwargs):
        super().__init__()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.optimizer = opt
        self.device = device
        self.criterion = crit
        self.pred_thresh = pred_thresh
        for k in kwargs:
            setattr(self, k, kwargs[k])
        if checkpoint:
            self.load_checkpoint(checkpoint, load_opt=load_opt, load_crit=load_crit, load_misc=load_misc)
    
    def forward_(self, x):
        if not is_list(x):
            x = [x]
        x = [self.model[0](i) for i in x]
        x = torch.cat(x, dim=1)
        return x

    def forward(self, x, meta=None):
        x = self.forward_(x)
        if meta is None:
            return self.model[1](x)
        # ftrs = torch.cat([flatten_tensor(self.model[0](x)), meta], dim=1)
        return self.model[1](x, meta=meta)
    
    def compute_loss(self, outputs, labels, class_weights=None, extra_loss_func=None, **kwargs):

        def compute_loss_(outputs, labels, class_weights=None):
            if extra_loss_func is not None:
                r = getattr(self.criterion, 'reduction')
                setattr(self.criterion, 'reduction', 'none')
                loss = self.criterion(outputs, labels)
                # print('ooooh')
                setattr(self.criterion, 'reduction', r)
                loss = extra_loss_func(loss)
                # print('focal')
                return loss
            if class_weights is not None:
                class_weights = class_weights.to(outputs.device)
                if is_bce(self.criterion):
                    r = getattr(self.criterion, 'reduction')
                    setattr(self.criterion, 'reduction', 'none')
                    loss = (self.criterion(outputs, labels) * class_weights).mean()
                    setattr(self.criterion, 'reduction', r)
                    return loss
                elif is_cross_entropy(self.criterion):
                    w = getattr(self.criterion, 'weight')
                    setattr(self.criterion, 'weight', class_weights)
                    loss = self.criterion(outputs, labels)
                    setattr(self.criterion, 'weight', w)
                    return loss
            return self.criterion(outputs, labels)
        
        if is_list(outputs):
            loss = 0
            for o,l in zip(outputs, labels):
                loss += compute_loss_(o, l, class_weights=class_weights)
        else:
            loss = compute_loss_(outputs, labels, class_weights=class_weights)
        return loss
    
    def open_batch(self, data_batch, device=None):
        device = default_device(device)
        inputs = data_batch['x']
        if is_list(inputs):       
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = inputs.to(device)
        labels = None
        if 'label' in data_batch.keys():
            labels = data_batch['label']
            if is_list(labels):
                labels = [l.to(device) for l in labels]
            else:
                labels = labels.to(device)
        meta = None
        if 'meta' in data_batch.keys():
            meta = data_batch['meta'].to(device)
        return {'inputs': inputs, 'labels': labels, 'meta': meta}

    def process_batch(self, data_batch, device=None):
        device = default_device(device)
        inputs, labels, meta = dict_values(self.open_batch(data_batch, device))

        # inputs = data_batch['x']
        # inputs = inputs.to(device)
        # labels = None
        # if 'label' in data_batch.keys():
        #     labels = data_batch['label']
        #     if is_list:
        #         labels = [l.to(device) for l in labels]
        #     else:
        #         labels = labels.to(device)
        # meta = None
        # if 'meta' in data_batch.keys():
        #     meta = data_batch['meta'].to(device)

        outputs = self.forward(inputs, meta=meta)
        return outputs, labels

    def batch_to_loss(self, data_batch, backward_step=True, class_weights=None, extra_loss_func=None, device=None, **kwargs):
        if device is None:
            device = self.device
        # inputs,labels = data_batch['x'], data_batch['label']
        # inputs = inputs.to(device)
        # labels = labels.to(device)
        # meta = None
        # if 'meta' in data_batch.keys():
        #     meta = data_batch['meta'].to(device)
        # outputs = self.forward(inputs, meta=meta)
        outputs, labels = self.process_batch(data_batch, device=device)
        loss = self.compute_loss(outputs, labels, class_weights=class_weights, extra_loss_func=extra_loss_func)
        if backward_step:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # ret = {'loss': loss.item(), 'outputs': outputs}
        return loss.item(), outputs
    
    def ss_forward(self, img1, img2):
        if len(list(self.model.children())) == 2:
            ss_model = copy.deepcopy(self.model[0])
            return flatten_tensor(ss_model(img1).detach()), flatten_tensor(self.model[0](img2))
        else:
            ss_model = copy.deepcopy(self.model)
            return flatten_tensor(ss_model(img1).detach()), flatten_tensor(self.model(img2))

    def ss_batch_to_loss(self, data_batch, backward_step=True, class_weights=None, extra_loss_func=None, device=None, **kwargs):
        if device is None:
            device = self.device
        # img1, labels, img2, x2 = data_batch[0], data_batch[1], data_batch[2], data_batch[3]
        img1, labels, img2, x2 = data_batch['x'], data_batch['label'], data_batch['ss_img'], data_batch['x2']
        img1 = img1.to(device)
        img2 = img2.to(device)
        meta = None
        x2 = x2.to(device)
        labels = labels.to(device)
        ss_outputs = self.ss_forward(img2, x2)
        if 'meta' in data_batch.keys():
            meta = data_batch['meta'].to(device)
        outputs = self.forward(img1, meta=meta)
        loss = self.compute_loss(outputs, labels, class_weights=class_weights, extra_loss_func=extra_loss_func)
        y = torch.ones(ss_outputs[0].shape[0]).to(device)
        # print(y.shape, ss_outputs[0].shape)
        l = torch.nn.CosineEmbeddingLoss()
        ss_loss = l(*ss_outputs, y)
        loss += ss_loss
        if backward_step:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        ret = {'loss': loss.item(), 'outputs': outputs}
        return loss.item(), outputs

    def update_accuracy(self, outputs, labels, classifier, metric, thresh=None):
        if thresh is None:
            thresh = tensor(self.pred_thresh).to(self.device)
        else:
            thresh = tensor(thresh).to(self.device)
        if metric == 'accuracy':
            classifier.update_accuracies(outputs, labels)
            # try:
                # y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                # _, preds = torch.max(torch.exp(outputs), 1)
                # y_pred.extend(list(preds.cpu().numpy()))
            # except:
                # pass
        elif metric == 'multi_accuracy':
            classifier.update_multi_accuracies(outputs, labels, thresh)

    def val_batch_to_loss(self, data_batch, metric='loss', thresh=None, class_weights=None, extra_loss_func=None, **kwargs):
        ret = {}
        # inputs,labels = data_batch['x'], data_batch['label']
        # inputs = inputs.to(self.device)
        # labels = labels.to(self.device)
        # meta = None
        # if 'meta' in data_batch.keys():
            # meta = data_batch['meta'].to(self.device)
        # outputs = self.forward(inputs, meta=meta)
        outputs, labels = self.process_batch(data_batch, device=self.device)
        loss = self.compute_loss(outputs, labels, class_weights=class_weights, extra_loss_func=extra_loss_func)
        ret['loss'] = loss.item()
        ret['outputs'] = outputs
        if 'accuracy' in metric:
            self.update_accuracy(outputs, labels, kwargs['classifier'], metric, thresh=thresh)
        
        return loss.item(), outputs

    def val_ss_batch_to_loss(self, data_batch, metric='loss', **kwargs):
        ret = {}
        device = self.device
        # img1, labels, img2, x2 = data_batch[0], data_batch[1], data_batch[2], data_batch[3]
        img1, labels, img2, x2 = data_batch['x'], data_batch['label'], data_batch['ss_img'], data_batch['x2']
        img1 = img1.to(device)
        img2 = img2.to(device)
        meta = None
        x2 = x2.to(device)
        labels = labels.to(device)
        ss_outputs = self.ss_forward(img2, x2)
        if 'meta' in data_batch.keys():
            meta = data_batch['meta'].to(device)
        outputs = self.forward(img1, meta=meta)
        loss = self.compute_loss(outputs, labels)
        y = torch.ones(ss_outputs[0].shape[0]).to(device)
        l = torch.nn.CosineEmbeddingLoss()
        ss_loss = l(*ss_outputs, y)
        loss += ss_loss
        ret['loss'] = loss.item()
        ret['outputs'] = outputs
        if 'accuracy' in metric:
            self.update_accuracy(outputs, labels, kwargs['classifier'], metric)
        return loss.item(), outputs

    def predict(self, x, actv=None, device=None):

        if device is None:
            device = self.device
    
        self.eval()
        self.model.eval()
        self.model = self.model.to(device)
        with torch.no_grad():
            # print(x.shape)
            if is_dict(x):
                outputs,_ = self.process_batch(x, device=device)
            else:
                if is_tensor(x):
                    if x.dim() == 3:
                        x.unsqueeze_(0)
                elif is_array(x):
                    x = to_tensor(x).unsqueeze(0)
                    # print(x)
                x = x.to(device)
                outputs = self.forward(x)
        if actv is not None:
            return actv(outputs)
        return outputs

    def freeze(self):
        freeze_params(params(self.model))
        modules = list(self.model.children())
        if len(modules) > 1:
            unfreeze_params(params(modules[-1]))
        else:
            for m in modules:
                if len(m) > 1:
                    unfreeze_params(params(m[-1]))
                    break

    def unfreeze(self):
        unfreeze_params(params(self.model))

    def checkpoint_dict(self):
        return {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'criterion': self.criterion, 'device': self.device, 'ped_thresh': self.pred_thresh}
    
    def save_checkpoint(self, save_name='model_checkpoint.pth', checkpoint_folder='dai_model_checkpoints'):
        checkpoint = self.checkpoint_dict()
        os.makedirs(checkpoint_folder, exist_ok=True)
        save_name = Path(checkpoint_folder)/save_name
        torch.save(checkpoint, save_name)

    def load_checkpoint(self, checkpoint, load_opt=True, load_crit=True, load_misc=True):

        if is_str(checkpoint) or is_path(checkpoint):
            checkpoint = torch.load(checkpoint)
        try:
            load_state_dict(self.model, checkpoint['model'])
            if load_opt:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model = self.model.to(self.device)
        except: pass
        for k in checkpoint.keys():
            if k not in ['model', 'optimizer']:
                if k == 'criterion':
                    if load_crit:
                        setattr(self, k, checkpoint[k])
                elif load_misc:
                    setattr(self, k, checkpoint[k])

class SimilarityModel(DaiModel):
    def __init__(self, model, opt, crit=nn.CosineEmbeddingLoss(), device=None, checkpoint=None, load_opt=False,
                 load_crit=False, load_misc=False):
        super().__init__(model=model, opt=opt, crit=crit, device=device, checkpoint=checkpoint, load_opt=load_opt,
                         load_crit=False, load_misc=False)
        # self.model = model.to(device)
        # self.optimizer = opt
        # self.device = device
        # self.criterion = crit
        # self.pred_thresh = pred_thresh

        # if checkpoint:
        #     self.load_checkpoint(checkpoint)

    def forward(self, x1, x2=None, mode=0):
        if mode == 0 and x2 is not None:
            return flatten_tensor(self.model(x1)), flatten_tensor(self.model(x2))
        elif mode == 1:
            return flatten_tensor(self.model(x1))
    
    # def forward(self, x1, x2):
        # ftrs = torch.cat([self.model.encoder(x1), self.model.encoder(x2)], dim=1)
        # return self.model.head(ftrs)
    
    def compute_loss(self, outputs, y):
        if list_or_tuple(outputs):
            return self.criterion(*outputs, y)
        return self.criterion(outputs, y)

    def process_batch(self, data_batch, device=None):
        if device is None:
            device = self.device
        # img1,img2 = data_batch[0], data_batch[1]
        img1,img2 = data_batch['x'], data_batch['x2']
        img1 = img1.to(device)
        img2 = img2.to(device)
        outputs = self.forward(img1, img2)
        return outputs

    def batch_to_loss(self, data_batch, backward_step=True, device=None, **kwargs):
        if device is None:
            device = self.device
        # img1,img2 = data_batch['x'], data_batch['x2']
        # img1 = img1.to(device)
        # img2 = img2.to(device)
        # outputs = self.forward(img1, img2)
        outputs = self.process_batch(data_batch, device=device)
        y = data_batch['same'][0] * torch.ones(outputs[0].shape[0]).to(device)
        # print(y.shape, outputs[0].shape)
        loss = self.compute_loss(outputs, y)
        if backward_step:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        ret = {'loss': loss.item(), 'outputs': outputs}
        return loss.item(), outputs

    def val_batch_to_loss(self, data_batch, metric='loss', **kwargs):
        ret = {}
        # img1,img2 = data_batch['x'], data_batch['x2']
        # img1 = img1.to(device)
        # img2 = img2.to(device)
        # outputs = self.forward(img1, img2)
        outputs = self.process_batch(data_batch)
        y = data_batch['same'][0] * torch.ones(outputs[0].shape[0]).to(self.device)
        loss = self.compute_loss(outputs, y)
        ret['loss'] = loss.item()
        ret['outputs'] = outputs
        # if 'accuracy' in metric:
            # self.update_accuracy(outputs, labels, kwargs['classifier'], metric)
        
        return loss.item(), outputs

    def predict(self, x, actv=None, device=None, **kwargs):

        if device is None:
            device = self.device
    
        self.eval()
        self.model.eval()
        self.model = self.model.to(device)
        with torch.no_grad():
            if is_dict(x):
                outputs = self.process_batch(x, device=device)
            else:
                for i in range(2):
                    # print(x.shape)
                    if is_tensor(x[i]):
                        if x[i].dim() == 3:
                            x[i].unsqueeze_(0)
                    elif is_array(x[i]):
                        x[i] = to_tensor(x[i]).unsqueeze(0)
                        # print(x)
                    x[i] = x[i].to(device)
                outputs = self.forward(*x)
        if actv is not None:
            return actv(outputs)
        return outputs

    def get_embeddings(self, x, device=None):
        self.eval()
        self.model.eval()
        if device is None:
            device = self.device
        x = x['x'].to(device)
        self.model = self.model.to(device)
        return flatten_tensor(self.model(x))

class MatchingModel(DaiModel):
    def __init__(self, model, opt, crit=nn.CrossEntropyLoss(), device=None, checkpoint=None, load_opt=False,
                 load_crit=False, load_misc=False):
        super().__init__(model=model, opt=opt, crit=crit, device=device, checkpoint=checkpoint, load_opt=load_opt,
                         load_crit=False, load_misc=False)
    def matcher(self, x1, x2):
        ftrs = torch.cat([(x1), (x2)], dim=1)
        return self.model[1](ftrs)

    def extractor(self, x1, x2=None, mode=0):
        if mode == 0 and x2 is not None:
            return self.model[0](x1), self.model[0](x2)
        elif mode == 1:
            return self.model[0](x1)

    def forward(self, x1, x2):
        return self.matcher(*self.extractor(x1, x2))
    
    def compute_loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def process_batch(self, data_batch, device=None):
        if device is None:
            device = self.device
        # img1,img2 = data_batch[0], data_batch[1]
        img1,img2 = data_batch['x'], data_batch['x2']
        img1 = img1.to(device)
        img2 = img2.to(device)
        outputs = self.forward(img1, img2)
        labels = data_batch['label']
        labels = labels.to(device)
        return outputs, labels

    def batch_to_loss(self, data_batch, backward_step=True, device=None, **kwargs):
        if device is None:
            device = self.device
        outputs, labels = self.process_batch(data_batch, device=device)
        loss = self.compute_loss(outputs, labels)
        if backward_step:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        ret = {'loss': loss.item(), 'outputs': outputs}
        return loss.item(), outputs

    def val_batch_to_loss(self, data_batch, metric='loss', **kwargs):
        ret = {}
        outputs, labels = self.process_batch(data_batch)
        loss = self.compute_loss(outputs, labels)
        ret['loss'] = loss.item()
        ret['outputs'] = outputs
        if 'accuracy' in metric:
            self.update_accuracy(outputs, labels, kwargs['classifier'], metric)
        
        return loss.item(), outputs

    def predict(self, x, actv=None, device=None):

        if device is None:
            device = self.device
    
        self.eval()
        self.model.eval()
        self.model = self.model.to(device)
        with torch.no_grad():
            for i in range(2):
                # print(x.shape)
                if is_tensor(x[i]):
                    if x[i].dim() == 3:
                        x[i].unsqueeze_(0)
                elif is_array(x[i]):
                    x[i] = to_tensor(x[i]).unsqueeze(0)
                    # print(x)
                x[i] = x[i].to(device)
            outputs = self.forward(*x)
        if actv is not None:
            return actv(outputs)
        return outputs

class DaiObjModel(DaiModel):
    def __init__(self, model, opt=None, device=None, checkpoint=None,
                 load_opt=False, load_misc=False, **kwargs):
        super().__init__(**locals_to_params(locals()))
    
    def extract_batch(self, data_batch, device=None):
        if device is None:
            device = self.device
        images, targets = data_batch
        images = list(image.to(device) for image in images)
        if self.model.training:
            targets = [{k:v for k,v in t.items() if is_tensor(v)} for t in targets]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        else:
            del targets
            targets = None
        return images, targets

    def batch_to_loss(self, data_batch, backward_step=True, device=None, **kwargs):
        if device is None:
            device = self.device
        images, targets = self.extract_batch(data_batch, device)

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        if backward_step:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item(), loss_dict
    
    def val_batch_to_loss(self, data_batch, metric='loss', **kwargs):
        # return self.batch_to_loss(data_batch, backward_step=False)
        cpu_device = torch.device('cpu')
        images, _ = self.extract_batch(data_batch)
        outputs = self.model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        return None, outputs

    def predict(self, img, score_thresh=0.5, class_names=None, device=None):
        device = default_device()
        self.model.eval().to(device)
        img = to_tensor(img)
        if not is_list(img):
            img = [img]
        img = [i.to(device) for i in img]
        preds = self.model(img)
        for i,pred in enumerate(preds):
            mask = pred['scores'] >= score_thresh
            preds[i]['boxes'] = pred['boxes'][mask].detach().cpu()
            preds[i]['labels'] = pred['labels'][mask].detach().cpu()
            preds[i]['scores'] = pred['scores'][mask].detach().cpu()
            if class_names is not None:
                preds[i]['labels'] = [class_names[l] for l in preds[i]['labels']]
        return preds