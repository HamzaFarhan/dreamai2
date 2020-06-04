from .utils import *
from .plot_eval import *
from .dai_imports import *

models_meta = {resnet34: {'cut': -2, 'conv_channels': 512},
               resnet50: {'cut': -2, 'conv_channels': 2048},
               densenet121: {'cut': -1, 'conv_channels': 1024}}

imagenet_stats = (tensor([0.485, 0.456, 0.406]), tensor([0.229, 0.224, 0.225]))

DEFAULTS = {'models_meta': models_meta, 'metrics': ['loss', 'accuracy', 'multi_accuracy'],
            'imagenet_stats': imagenet_stats, 'image_extensions': image_extensions}

def create_body(arch, pretrained=True, cut=None, num_extra=3):
    model = arch(pretrained=pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    modules = list(model.children())[:cut]
    channels = models_meta[arch]['conv_channels']
    extra_convs = [conv_block(channels, channels)]*num_extra
    modules += extra_convs
    return nn.Sequential(*modules)

def create_head(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True,
                bn_final=False, lin_first=False, y_range=None):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    ps = [ps]
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += LinBnDrop(ni, no, bn=True, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

def create_model(arch, num_classes, num_extra=3):
    meta = models_meta[arch]
    body = create_body(arch, meta['cut'], num_extra=num_extra)
    head = create_head(meta['conv_channels']*2, num_classes)
    net = nn.Sequential(body, head)
    return net

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
        self.class_correct = defaultdict(int)
        self.class_totals = defaultdict(int)

    def update_accuracies(self,outputs,labels):
        _, preds = torch.max(torch.exp(outputs), 1)
        # _, preds = torch.max(outputs, 1)
        correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
        for i in range(labels.shape[0]):
            label = labels.data[i].item()
            self.class_correct[label] += correct[i].item()
            self.class_totals[label] += 1

    def fai_update_multi_accuracies(self, preds, label):
        correct = label*preds
        class_idx = torch.nonzero(label)[0]
        for idx in class_idx:
            c = correct[idx].item()
            idx = idx.item()
            self.class_correct[idx] += c
            self.class_totals[idx] += 1

    def update_multi_accuracies(self,outputs,labels,thresh=0.5):
        preds = torch.sigmoid(outputs) > thresh
        correct = (labels==1)*(preds==1)
        for i in range(labels.shape[0]):
            label = torch.nonzero(labels.data[i]).squeeze(1)
            for l in label:
                c = correct[i][l].item()
                l = l.item()
                self.class_correct[l] += c
                self.class_totals[l] += 1

    def get_final_accuracies(self):
        accuracy = (100*np.sum(list(self.class_correct.values()))/np.sum(list(self.class_totals.values())))
        try:
            class_accuracies = [(self.class_names[i],100.0*(self.class_correct[i]/self.class_totals[i])) 
                                 for i in self.class_names.keys() if self.class_totals[i] > 0]
        except:
            class_accuracies = [(self.class_names[i],100.0*(self.class_correct[i]/self.class_totals[i])) 
                                 for i in range(len(self.class_names)) if self.class_totals[i] > 0]
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
    def __init__(self, model, opt, crit=nn.BCEWithLogitsLoss(), pred_thresh=0.5,
                 device='cpu', checkpoint=None, load_opt=False):
        super().__init__()
        self.model = model.to(device)
        self.optimizer = opt
        self.device = device
        self.criterion = crit
        self.pred_thresh = pred_thresh

        if checkpoint:
            self.load_checkpoint(checkpoint, load_opt=load_opt)
    
    def forward(self, x):
        return self.model(x)
    
    def compute_loss(self, outputs, labels):
        return self.criterion(outputs, labels)
    
    def process_batch(self, data_batch):
        inputs,labels = data_batch[0], data_batch[1]
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.forward(inputs)
        return outputs

    def batch_to_loss(self, data_batch, backward_step=True, device=None):
        if device is None:
            device = self.device
        inputs,labels = data_batch[0], data_batch[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs,labels)
        if backward_step:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        ret = {'loss': loss.item(), 'outputs': outputs}
        return loss.item(), outputs
    
    def ss_forward(self, img1, img2):
        ss_model = copy.deepcopy(self.model[0])
        return flatten_tensor(ss_model(img1).detach()), flatten_tensor(self.model[0](img2))

    def ss_batch_to_loss(self, data_batch, backward_step=True, device=None):
        if device is None:
            device = self.device
        img1, labels, img2, x2 = data_batch[0], data_batch[1], data_batch[2], data_batch[3]
        img1 = img1.to(device)
        img2 = img2.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        ss_outputs = self.ss_forward(img2, x2)
        outputs = self.forward(img1)
        loss = self.compute_loss(outputs, labels)
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

    def update_accuracy(self, outputs, labels, classifier, metric):
        if metric == 'accuracy':
            classifier.update_accuracies(outputs, labels)
            # try:
                # y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                # _, preds = torch.max(torch.exp(outputs), 1)
                # y_pred.extend(list(preds.cpu().numpy()))
            # except:
                # pass
        elif metric == 'multi_accuracy':
            classifier.update_multi_accuracies(outputs, labels, self.pred_thresh)

    def val_batch_to_loss(self, data_batch, metric='loss', **kwargs):
        ret = {}
        inputs,labels = data_batch[0], data_batch[1]
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, labels)
        ret['loss'] = loss.item()
        ret['outputs'] = outputs
        if 'accuracy' in metric:
            self.update_accuracy(outputs, labels, kwargs['classifier'], metric)
        
        return loss.item(), outputs

    def val_ss_batch_to_loss(self, data_batch, metric='loss', **kwargs):
        ret = {}
        device = self.device
        img1, labels, img2, x2 = data_batch[0], data_batch[1], data_batch[2], data_batch[3]
        img1 = img1.to(device)
        img2 = img2.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)
        ss_outputs = self.ss_forward(img2, x2)
        outputs = self.forward(img1)
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
    
    def save_checkpoint(self, save_name='model_checkpoint.pth'):
        checkpoint = self.checkpoint_dict()
        torch.save(checkpoint, save_name)

    def load_checkpoint(self, checkpoint, load_opt=True):

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
                setattr(self, k, checkpoint[k])

class SimilarityModel(DaiModel):
    def __init__(self, model, opt, crit=nn.CosineEmbeddingLoss(), device='cpu', checkpoint=None, load_opt=False):
        super().__init__(model=model, opt=opt, crit=crit, device=device, checkpoint=checkpoint, load_opt=load_opt)
        # self.model = model.to(device)
        # self.optimizer = opt
        # self.device = device
        # self.criterion = crit
        # self.pred_thresh = pred_thresh

        # if checkpoint:
        #     self.load_checkpoint(checkpoint)
    
    def forward(self, x1, x2):
        return self.model(x1), self.model(x2)
    
    # def forward(self, x1, x2):
        # ftrs = torch.cat([self.model.encoder(x1), self.model.encoder(x2)], dim=1)
        # return self.model.head(ftrs)
    
    def compute_loss(self, outputs, y):
        return self.criterion(outputs, y)
    
    def process_batch(self, data_batch):
        inputs,labels = data_batch[0], data_batch[1]
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.forward(inputs)
        return outputs

    def batch_to_loss(self, data_batch, backward_step=True, device=None):
        if device is None:
            device = self.device
        img1,img2 = data_batch[0], data_batch[1]
        img1 = img1.to(device)
        img2 = img2.to(device)
        outputs = self.forward(img1, img2)
        y = torch.ones(ss_outputs[0][0].shape[0]).to(device)
        loss = self.compute_loss(*outputs, y)
        if backward_step:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        ret = {'loss': loss.item(), 'outputs': outputs}
        return loss.item(), outputs

    def val_batch_to_loss(self, data_batch, metric='loss', **kwargs):
        ret = {}
        img1,img2 = data_batch[0], data_batch[1]
        img1 = img1.to(device)
        img2 = img2.to(device)
        outputs = self.forward(img1, img2)
        y = torch.ones(ss_outputs[0][0].shape[0]).to(device)
        loss = self.compute_loss(*outputs, y)
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

