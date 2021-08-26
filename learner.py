from .obj import *
from .core import *

class GetAttr:
    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default='default'
    def _component_attr_filter(self,k):
        if k.startswith('__') or k in ('_xtra',self._default): return False
        xtra = getattr(self,'_xtra',None)
        return xtra is None or k in xtra
    def _dir(self): return [k for k in dir(getattr(self,self._default)) if self._component_attr_filter(k)]
    def __getattr__(self,k):
        if self._component_attr_filter(k):
            attr = getattr(self,self._default,None)
            if attr is not None: return getattr(attr,k)
        raise AttributeError(k)
    def __dir__(self): return custom_dir(self,self._dir())
#     def __getstate__(self): return self.__dict__
    def __setstate__(self,data): self.__dict__.update(data)

class Callback(GetAttr):
    _default = 'learner'

    def before_fit(self): pass
    
    def before_training(self): pass
    def before_train_epoch(self): pass
    def after_train_epoch(self): pass
    def before_train_batch(self): pass
    def after_train_batch(self): pass
    def after_training(self): pass
    
    def before_valid(self): pass
    def before_val_epoch(self): pass
    def after_val_epoch(self): pass
    def before_val_batch(self): pass
    def after_val_batch(self): pass
    def after_valid(self): pass
    
    def after_fit(self): pass

    def before_predict(self): pass
    def after_predict(self): pass

def pushbullet_message(body, title='Update',
                       token='o.9tvOgIgF7AEurn6imXKpeed4CZpQxOE7'):
    msg = {"type": "note", "title": title, "body": body}
    TOKEN = 'PUSHBULLET_TOKEN'
    resp = requests.post('https://api.pushbullet.com/v2/pushes', 
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + token,
                                  'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('Error',resp.status_code)
    # else:
    #     print ('Message sent') 

class CheckpointCallback(Callback):
    def __init__(self, metric='loss', curr_best=None, send_text=False,
                 save_name='model_checkpoint', best_name='best_model',
                 checkpoint_folder = 'dai_model_checkpoints'):

        self.send_text = send_text
        self.save_metric = metric
        # self.save_every = save_every
        if curr_best is None:
            if metric == 'loss':
                self.curr_best = 100.
            else:
                self.curr_best = 0.
        else:
            self.curr_best = curr_best
        if Path(save_name).suffix not in ['.pkl','.pth','.pt']:
            save_name = save_name+'.pth'
        if Path(best_name).suffix not in ['.pkl','.pth','.pt']:
            best_name = best_name+'.pth'
        # checkpoint_folder = 'dai_model_checkpoints'
        os.makedirs(checkpoint_folder, exist_ok=True)
        self.save_name = Path(checkpoint_folder)/save_name
        self.best_name = Path(checkpoint_folder)/best_name

    def checker(self, curr, best, metric='loss'):
        if metric == 'loss':
            return curr <= best
        return curr >= best

    def before_fit(self):
        self.not_improved = 0

    def get_curr_metric(self):
        if self.save_class is not None:
            class_acc = self.val_ret['class_accuracies']
            for ca in class_acc:
                if self.save_class in ca:
                    curr_metric = ca[1]
                    # print(ca)
                    break
        else:
            curr_metric = self.val_ret[self.save_metric]
        return curr_metric

    def after_valid(self):
        curr_metric = self.get_curr_metric()
        if self.trial is not None:
            self.trial.report(curr_metric, self.curr_epoch)

            # Handle pruning based on the intermediate value.
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        self.learner.curr_metric = curr_metric
        if (self.save_every is not None) and ((self.curr_epoch+1) % self.save_every == 0): 
            checkpoint = self.model.checkpoint_dict()
            checkpoint[self.save_metric] = curr_metric
            if 'accuracy' in self.save_metric:
                checkpoint['class_names'] = self.dls.class_names
            torch.save(checkpoint, self.save_name.parent/(self.save_name.stem+f'_{curr_metric}'+self.save_name.suffix))

        if self.checker(curr_metric, self.curr_best, self.save_metric) and self.save_best:
            self.not_improved = 0
            if self.print_progress:
                top = f'\n**********Updating best {self.save_metric}**********\n'
                print(top)
                print(f'Previous best: {self.curr_best:.5f}')
                print(f'New best: {curr_metric:.5f}\n')
                bottom = '*'*(len(top)-2)
                print(f'{bottom}\n')
                if self.send_text:
                    msg = f'{top}\n\nPrevious best: {self.curr_best:.5f}\nNew best: {curr_metric:.5f}\n\n{bottom}'
                    pushbullet_message(msg)
            self.curr_best = curr_metric
            checkpoint = self.model.checkpoint_dict()
            checkpoint[self.save_metric] = self.curr_best
            if 'accuracy' in self.save_metric:
                checkpoint['class_names'] = self.dls.class_names
            # torch.save(checkpoint, self.best_name.parent/(self.best_name.stem+f'_{curr_metric}'+self.best_name.suffix))
            torch.save(checkpoint, self.best_name)

        elif self.early_stopping_epochs is not None:
            self.not_improved += 1
            if self.not_improved >= self.early_stopping_epochs:
                self.learner.do_training = False
                self.learner.load_best = True
                print('+----------------------------------------------------------------------+')
                print(' Early Stopping.')
                print('+----------------------------------------------------------------------+')
    def after_fit(self):
        self.learner.curr_metric = self.curr_metric
        if self.best_name.exists() and self.load_best and self.trial is None:
            checkpoint = torch.load(self.best_name)
            self.learner.checkpoint = checkpoint
            self.model.load_checkpoint(checkpoint)
            # load_state_dict(self.model, checkpoint['model'])
            # self.learner.model.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Best model loaded.')

class BasicCallback(Callback):
    
    def before_fit(self):
        self.learner.total_time = 0
    
    def before_training(self):
        self.learner.model.train()
        self.learner.t0 = time.time()
    
    def before_train_epoch(self):
        self.learner.tr_running_loss = 0.
    
    def before_train_batch(self):
        self.learner.t1 = time.time()
    
    def after_train_batch(self):
        self.learner.tr_running_loss += self.tr_batch_loss    
    
    def after_train_epoch(self):
        self.learner.tr_ret = {}
        self.learner.tr_ret['loss'] = self.tr_running_loss/self.num_batches        
        
    def before_valid(self):
        self.learner.model.eval()
        self.learner.t2 = time.time()
        
    def before_val_epoch(self):
        self.learner.val_running_loss = 0.
        self.learner.classifier = None
        if 'accuracy' in self.learn_metric:
            self.learner.classifier = Classifier(self.dls.class_names)
    
    def before_val_batch(self):
        if not hasattr(self.learner, 'classifier'):
            self.learner.classifier = None

    def after_val_batch(self):
        if self.val_batch_loss is not None:
            self.learner.val_running_loss += self.val_batch_loss
    
    def after_val_epoch(self):
        self.learner.val_ret = {}
        if self.val_batch_loss is not None:
            self.learner.val_ret['loss'] = self.val_running_loss/self.num_batches
        if 'accuracy' in self.learn_metric:
            # self.learner.val_ret[self.learn_metric],\
            # self.learner.val_ret['class_accuracies'] = self.classifier.get_final_accuracies()
            acc, class_acc = self.classifier.get_final_accuracies()
            self.learner.val_ret[self.learn_metric] = acc
            if is_list(acc):
                self.learner.val_ret[f'mean {self.learn_metric}'] = torch.as_tensor(acc).mean().item()
                # print(self.learner.val_ret[f'mean {self.learn_metric}'])
            self.learner.val_ret['class_accuracies'] = class_acc

    def after_valid(self):
        self.learner.model.train()

    def after_predict(self):
        if self.learn_metric == 'accuracy':
            p = nn.Softmax()(self.pred_out).cpu()
            m = torch.max(p, 1)
            c = np.array(self.dls.class_names)[m[1]]
            # pred_out = []
            p = p[0]
            self.learner.pred_out = {'probs': p, 'pred': c}
            # for i,pred in enumerate(p):
            #     if is_str(c):
            #         pred_out.append([pred, c])
            #     else:
            #         pred_out.append([pred, c[i]])
            # self.learner.pred_out = flatten_list(pred_out)
            # self.learner.pred_out = [p.tolist(), c.tolist()]

        elif self.learn_metric == 'multi_accuracy':
            p = nn.Sigmoid()(self.pred_out).cpu()[0]
            bools = (p >= torch.as_tensor(self.pred_thresh))
            # print(bools)
            c = np.array(self.dls.class_names)[bools]
            self.learner.pred_out = {'probs': p, 'bools':bools, 'pred': c}
        # if 'accuracy' in self.learn_metric:

class ObjCallback(BasicCallback):    
    def __init__(self, obj_output='obj_outputs'):
        super().__init__()
        self.obj_output = obj_output

    def before_val_epoch(self):
        coco = get_coco_api_from_dataset(self.dls.valid_ds)
        iou_types = get_iou_types(self.model.model)
        self.learner.coco_evaluator = CocoEvaluator(coco, iou_types)

    def after_val_batch(self):
        _, targets = self.data_batch
        res = {target["image_id"].item(): output for target, output in zip(targets, self.val_batch_out)}
        self.coco_evaluator.update(res)

    def after_val_epoch(self):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        self.learner.val_ret = self.coco_evaluator.coco_eval['bbox'].metric_dict
        # self.learner.val_ret['MAP'] = self.val_running_loss/self.num_batches

def cyclical_lr(stepsize, min_lr=3e-2, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

class Ensemble():
    def __init__(self, nets, model_paths, model_class=DaiModel, model_weights=1., pred_thresh=0.5,
                 metric='accuracy', device=None, extra_loss_func=None):

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if path_or_str(model_paths):
            model_paths = Path(model_paths)
            model_paths = sorted_paths(model_paths)
        self.model_paths = model_paths
        if model_weights is None:
            model_weights = 1.
        if not list_or_tuple(model_weights):
            model_weights = [model_weights]*len(model_paths)
        self.model_weights = model_weights
        store_attr(self, 'pred_thresh,metric,device')
        if not list_or_tuple(nets):
            nets = [nets]*len(model_paths)
        self.models = [model_class(copy.deepcopy(net), device=device, checkpoint=torch.load(mp, map_location='cpu'),
                                   load_opt=True, load_crit=True, load_misc=False) for net,mp in zip(nets,model_paths)]
        self.extra_loss_func = extra_loss_func
        
    def batch_to_loss(self, data_batch, model_weights=None, class_weights=None, extra_loss_func=None, device=None):
        if model_weights is None:
            model_weights = self.model_weights
        if device is None:
            device = self.device
        if extra_loss_func is None:
            extra_loss_func = self.extra_loss_func
        if not list_or_tuple(model_weights):
            model_weights = [model_weights]*len(self.models)
        outputs_labels = [model.process_batch(data_batch, device=device) for model in self.models]
        # o = [ol[0] for ol in outputs_labels]
        outputs = torch.stack([ol[0]*mw for ol,mw in zip(outputs_labels, model_weights)])
        # print(outputs[1])
        outputs = torch.sum(outputs, dim=0)
        # print(outputs.shape)
        labels = outputs_labels[0][1]
        loss = self.models[0].compute_loss(outputs, labels, class_weights=class_weights, extra_loss_func=extra_loss_func)
        return loss.item(), outputs, labels
    
    def to_eval(self):
        for i in range(len(self.models)):
            self.models[i].eval()
    
    def to_train(self):
        for i in range(len(self.models)):
            self.models[i].train()

    def evaluate(self, dl, model_weights=None, class_weights=None, class_names=None,
                 metric=None, pred_thresh=None, device=None, extra_loss_func=None):

        n_batches = num_batches(dl)
        if device is None:
            device = self.device

        if pred_thresh is None:
            pred_thresh = self.pred_thresh

        if metric is None:
            metric = self.metric
        
        if model_weights is None:
            model_weights = self.model_weights
        
        if extra_loss_func is None:
            extra_loss_func = self.extra_loss_func
    
        running_loss = 0.
        classifier = None

        if 'accuracy' in metric:
            assert(class_names is not None), print("You must pass class_names for a classifier's evaluation.")
            classifier = Classifier(class_names)

        y_pred = []
        y_prob = []
        y_true = []

        self.to_eval()
        # self.model.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dl:
                loss, outputs, labels = self.batch_to_loss(data_batch=data_batch, model_weights=model_weights,
                                                           class_weights=class_weights, device=device, extra_loss_func=extra_loss_func)
                running_loss += loss
                if classifier is not None:
                    # labels = data_batch[1].to(device)
                    labels = data_batch['label'].to(device)
                    if metric == 'accuracy':
                        classifier.update_accuracies(outputs, labels)
                        try:
                            y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                            prob, preds = torch.max(nn.Softmax()(outputs).cpu(), 1)
                            y_pred.extend(list(preds.cpu().numpy()))
                            y_prob.extend(list(prob.cpu().numpy()))
                        except:
                            pass
                    elif metric == 'multi_accuracy':
                        classifier.update_multi_accuracies(outputs, labels, pred_thresh)
                        try:
                            y_true += (list(labels.cpu().numpy()))
                            prob = nn.Sigmoid()(outputs).cpu()
                            y_prob += (list(prob.cpu().numpy()))
                        except:
                            pass
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()

        self.to_train()

        ret = {}
        # print('Running_loss: {:.5f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.5f}'.format(rmse_))
            ret['final_rmse'] = rmse_/n_batches

        ret['final_loss'] = running_loss/n_batches

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            try:
                ret['report'] = ClassificationReport(classification_report(y_true, y_pred,
                                                            target_names=class_names),
                                                            ret['accuracy'], ret['class_accuracies'])
                ret['confusion_matrix'] = ConfusionMatrix(confusion_matrix(y_true, y_pred), class_names)
            except:
                print('Classification report and confusion matrix not available.')
                ret['report'] = None
                ret['confusion_matrix'] = None
            try:
                yt = np.array(y_true)
                yp = np.array(y_prob)
                ret['roc_auc_score'] = roc_auc_score(yt, yp, multi_class='ovo', labels=class_names)
            except:
                print('ROC AUC score not available.')
                ret['roc_auc_score'] = None
        return ret

    def predict(self, x, dl=None, model_weights=None, pred_thresh=None, metric=None, device=None):

        if pred_thresh is None:
            pred_thresh = self.pred_thresh

        if device is None:
            device = self.device

        if model_weights is None:
            model_weights = self.model_weights

        if metric is None:
            metric = self.metric

        if is_df(x):
            x = list(x.iloc[:,0])
        elif is_array(x):
            if x.ndim == 3:
                x = [x]
            else:
                x = list(x)
        elif is_tensor(x):
            x = tensor_to_img(x)
            if is_array(x):
                x = [x]
        if dl is not None:
            dset = dl.dataset
            tfms = dset.tfms
            # tfms = None
            pred_set = PredDataset(x, tfms=tfms)
        else:
            pred_set = x
        # pred_dl = DataLoader(self.pred_set, batch_size=bs)
        self.pred_outs = []
        for idx,data_batch in enumerate(pred_set):
            pred_out = self.models_predict(self.models, data_batch, model_weights=model_weights, device=device)
            if metric == 'accuracy':
                p = nn.Softmax()(pred_out).cpu()
                m = torch.max(p, 1)
                # c = np.array(self.dls.class_names)[m[1]]
                p = p[0]
                pred_out = {'probs': p, 'pred': m[1]}

            elif metric == 'multi_accuracy':
                p = nn.Sigmoid()(pred_out).cpu()[0]
                bools = (p >= torch.as_tensor(pred_thresh))
                # print(bools)
                c = np.array(self.dls.class_names)[bools]
                pred_out = {'probs': p, 'bools':bools}

            self.pred_outs.append(pred_out)

        return self.pred_outs

    def models_predict(self, models, data_batch, model_weights=None, device=None):
        if model_weights is None:
            model_weights = self.model_weights
        if not list_or_tuple(model_weights):
            model_weights = [model_weights]*len(models)
        outputs = [model.predict(data_batch, device=device) for model in models]
        outputs = torch.stack([ol[0]*mw for ol,mw in zip(outputs, model_weights)])
        outputs = torch.sum(outputs, dim=0).unsqueeze(0)
        return outputs

class Learner:
    def __init__(self, model, dls, model_splitter=None, metric='loss', cbs=[BasicCallback()], trial=None):

        store_attr(self, 'model,dls,model_splitter,cbs')
        self.learn_metric = metric
        if metric == 'loss':
            self.optuna_opt = 'minimize'
        else:
            self.optuna_opt = 'maximize'
        self.trial = trial
        for cb in cbs: cb.learner = self
        try:
            self.model.normalize = dls.normalize
            self.model.denorm = dls.denorm
            self.model.img_mean = dls.img_mean
            self.model.img_std = dls.img_std
        except:
            pass
        self.extra_loss_func = None
        self.is_frozen = False
        assert len(cbs) > 0, print('Please pass some callbacks for training.')
    
    def print_train_progress(self, progress={}):
        
        if (self.batch_num+1) % self.fit_print_every == 0:
            elapsed = time.time()-self.t0
            if elapsed > 60:
                elapsed /= 60.
                measure = 'min'
            else:
                measure = 'sec'
            batch_time = time.time()-self.t1

            if self.total_time == 0:
                # total_time = (batch_time*self.num_batches*self.fit_epochs) + (self.fit_epochs/self.fit_validate_every)
                # total_time += total_time/5
                total_time = (batch_time*self.num_batches)
                total_time += total_time/4
                if total_time > 60:
                    total_time /= 60.
                    total_measure = 'min'
                else:
                    total_measure = 'sec'
                self.total_time = total_time
                self.total_measure = total_measure
            total_time, total_measure = self.total_time, self.total_measure                

            if batch_time > 60:
                batch_time /= 60.
                measure2 = 'min'
            else:
                measure2 = 'sec'
            self.t1 = time.time()

            print('+----------------------------------------------------------------------+')
            print(f" {time.asctime().split()[-2]}")
            # print(f" Time elapsed: {elapsed:.5f}{measure} / {total_time:.5f}{total_measure}")
            print(f" Time elapsed: {elapsed:.5f} {measure}")
            print(f" Epoch:{self.curr_epoch+1}/{self.fit_epochs}")
            print(f" Batch: {self.batch_num+1}/{self.num_batches}")
            print(f" Batch training time: {batch_time:.5f} {measure2}")
            print(f" Batch training loss: {self.tr_batch_loss:.5f}")
            print(f" Average training loss: {self.tr_running_loss/(self.batch_num+1):.5f}")
            if len(progress) > 0:
                prog_keys = list(progress.keys())
                for k in prog_keys[:-1]:
                    print(f" {k}: {progress[k]:.6}\n")
                k  = prog_keys[-1]
                print(f" {k}: {progress[k]:.6}")
            print('+----------------------------------------------------------------------+')

    def print_valid_progress(self):
        tr_progress = self.tr_ret
        val_progress = self.val_ret
        time_elapsed = time.time() - self.t2
        if time_elapsed > 60:
            time_elapsed /= 60.
            measure = 'min'
        else:
            measure = 'sec'
        
        print('\n'+'/'*36)
        print(f"{time.asctime().split()[-2]}")
        print(f"Epoch {self.curr_epoch+1}/{self.fit_epochs}")
        print(f"Validation time: {time_elapsed:.6f} {measure}")

        if len(tr_progress) > 0:
            prog_keys = list(tr_progress.keys())
            for k in prog_keys:
                print(f"Epoch training {k}: {tr_progress[k]:.6}")

        if len(val_progress) > 0:
            prog_keys = list(val_progress.keys())
            for k in prog_keys:
                v = val_progress[k]
                if is_list(v):
                    v = v[:self.num_progress_classes]
                    # if len(v) <= 10:
                    if not (is_list(v[0]) and ((not self.verbose) or (self.curr_epoch != (self.fit_epochs-1)))):
                        print(f"Epoch validation {k}:")
                        for x in v:
                            print(f'    {x}')
                else:
                    print(f"Epoch validation {k}: {v:.6}")

        print('\\'*36+'\n')
    
    def train_batch(self):
        self('before_train_batch')
        if self.semi_sup:
            self.tr_batch_loss, self.tr_batch_out = self.model.ss_batch_to_loss(self.data_batch,
                                                               class_weights=self.class_weights[0],
                                                               extra_loss_func=self.extra_loss_func)
        else:
            self.tr_batch_loss, self.tr_batch_out = self.model.batch_to_loss(self.data_batch,
                                                               class_weights=self.class_weights[0],
                                                               extra_loss_func=self.extra_loss_func)
        if self.fit_scheduler is not None:
            self.fit_scheduler.step()
            # print(get_lr(self.model.optimizer))
        self('after_train_batch')
    
    def val_batch(self):
        self('before_val_batch')
        self.val_batch_loss, self.val_batch_out = self.model.val_batch_to_loss(self.data_batch, metric=self.learn_metric,
                                                  classifier=self.classifier, thresh=self.pred_thresh,
                                                  class_weights=self.class_weights[-1],
                                                  extra_loss_func=self.extra_loss_func)
        self('after_val_batch')
    
    def train_epoch(self):
        self('before_train_epoch')
        dl = self.dls.train
        # self.num_batches = len(dl)
        self.num_batches = num_batches(dl)
        for self.batch_num, self.data_batch in enumerate(dl):
        # for self.batch_num, self.data_batch in zip(range(self.num_batches),dl):
            self.train_batch()
            if self.print_progress:
                self.print_train_progress()
        del self.data_batch
        self('after_train_epoch')
    
    def val_epoch(self):
        self('before_val_epoch')
        dl = self.dls.valid
        # self.num_batches = len(dl)
        self.num_batches = num_batches(dl)
        with torch.no_grad():
            for self.batch_num, self.data_batch in enumerate(dl):
            # for self.batch_num, self.data_batch in zip(range(self.num_batches),dl):
                self.val_batch()
                del self.data_batch
        self('after_val_epoch')
        if self.print_progress:
            self.print_valid_progress()
    
    def fit(self, epochs, lr=None, metric='loss', print_every=3, validate_every=1, print_progress=True, save_every=None,
            load_best=True, semi_sup=False, early_stopping_epochs=None, cycle_len=0, save_class=None, pred_thresh=None,
            class_weights=None, save_best=True, progressive_resizing=None, extra_loss_func=None, verbose=True,
            fit_scheduler=None, num_progress_classes=10):
        
        self.fit_epochs = epochs
        self.learn_metric = metric
        self.fit_print_every = print_every
        self.fit_validate_every = validate_every
        self.load_best = load_best
        self.semi_sup = semi_sup
        self.early_stopping_epochs = early_stopping_epochs
        self.do_training = True
        self.fit_scheduler = fit_scheduler
        self.save_class = save_class
        self.num_progress_classes = num_progress_classes
        self.print_progress = print_progress
        self.verbose = verbose
        self.pred_thresh = pred_thresh
        self.save_every = save_every
        self.save_best = save_best
        self.progressive_resizing = progressive_resizing
        if not list_or_tuple(class_weights):
            class_weights = [class_weights]
        self.class_weights = class_weights
        if pred_thresh is None:
            self.pred_thresh = self.model.pred_thresh
        self.extra_loss_func = extra_loss_func

        def do_fit():
            if cycle_len > 0:
                cyclic_step = int((cycle_len/2)*len(self.dls.train))
                factor = 6
                set_lr(self.model.optimizer, 1.)
                end_lr = self.find_clr(self.dls.train, plot=False)
                clr = cyclical_lr(cyclic_step, min_lr=end_lr/factor, max_lr=end_lr)
                scheduler = optim.lr_scheduler.LambdaLR(self.model.optimizer, [clr])
                self.fit_scheduler = scheduler
            elif lr is not None:
                if list_or_tuple(lr):
                    lrs = lr
                    if len(self.model.optimizer.param_groups) == len(lrs):
                        for i in range(len(self.model.optimizer.param_groups)):
                            self.model.optimizer.param_groups[i]['lr'] = lrs[i]
                    else:
                        del self.model.optimizer.param_groups[:]
                        if self.model_splitter is not None:
                            p = self.model_splitter(self.model.model)
                        if len(lrs) != len(p):
                            p = split_params(self.model.model, len(lrs))
                        for param,lr_ in zip(p,lrs):
                            p_group = {'params': param, 'lr': lr_}
                            self.model.optimizer.add_param_group(p_group)
                else:
                    set_lr(self.model.optimizer, lr)

            if is_lars(self.model.optimizer):
                self.model.optimizer.epoch = 0
                self.model.optimizer.param_groups[0]['max_epoch'] = epochs

            self('before_fit')
            for self.curr_epoch in range(epochs):
                if not self.do_training:
                    break
                self('before_training')
                self.train_epoch()
                self('after_training')
                
                if validate_every and (self.curr_epoch % validate_every == 0):
                    self('before_valid')
                    self.val_epoch()
                    self('after_valid')                
            self('after_fit')

        if progressive_resizing is not None:
            for pr in progressive_resizing:
                if not list_or_tuple(pr):
                    pr = [pr]
                h,w = pr[0],pr[0]
                bs = None
                if len(pr) > 1:
                    bs = pr[1]
                if len(pr) > 2:
                    lr = pr[2]
                # set_lr(self.model.optimizer, p_lr)
                self.dls.progressive_resize(h=h, w=w, bs=bs)
                print(f'\nImage Shape: ({h},{w}), Batch Size: {self.dls.train.batch_size}\n')
                do_fit()
        else:
            do_fit()
        if self.trial is not None:
            return self.curr_metric

    def predict(self, x, pred_dset=PredDataset, pred_thresh=None, device=None, meta_dix=None,
                do_tta=False, tta=None, num_tta=3, **kwargs):

        if pred_thresh is None:
            self.pred_thresh = self.model.pred_thresh
        else:
            self.pred_thresh = pred_thresh

        # if is_df(x):
            # x = list(x.iloc[:,0])
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
        dl = self.dls.test
        dset = dl.dataset
        tfms = dset.tfms

        if tta is not None and not list_or_tuple(tta):
            tta = [tta]*num_tta
        # tfms = None
        self.pred_set = pred_dset(x, tfms=tfms, meta_idx=meta_dix, do_tta=do_tta, tta=tta, num_tta=num_tta, **kwargs)
        # pred_dl = DataLoader(self.pred_set, batch_size=bs)
        self.pred_outs = []
        for idx,data_batch in enumerate(self.pred_set):
            self('before_predict')
            if list_or_tuple(data_batch):
                for db in data_batch:
                    batchify_dict(db)
                self.pred_out = self.tta_predict(data_batch, device=device)
            else:
                batchify_dict(data_batch)
                self.pred_out = self.model.predict(data_batch, device=device)
            self('after_predict')
            self.pred_outs.append(self.pred_out)

        return self.pred_outs

    # def predict(self, x,pred_thresh=None, device=None, **kwargs):

    #     if pred_thresh is None:
    #         self.pred_thresh = self.model.pred_thresh
    #     else:
    #         self.pred_thresh = pred_thresh

    #     if is_df(x):
    #         x = list(x.iloc[:,0])
    #     elif is_array(x):
    #         if x.ndim == 3:
    #             x = [x]
    #         else:
    #             x = list(x)
    #     elif is_tensor(x):
    #         x = tensor_to_img(x)
    #         if is_array(x):
    #             x = [x]
    #     dl = self.dls.test
    #     dset = dl.dataset
    #     tfms = dset.tfms
    #     # tfms = None
    #     self.pred_set = PredDataset(x, tfms=tfms, **kwargs)
    #     # pred_dl = DataLoader(self.pred_set, batch_size=bs)
    #     self.pred_outs = []
    #     for idx,data_batch in enumerate(self.pred_set):
    #         self.pred_out = self.model.predict(data_batch, device=device)
    #         self('after_predict')
    #         self.pred_outs.append(self.pred_out)

    #     return self.pred_outs

    def tta_batch_to_loss(self, data_batches, class_weights=None, device=None):
        loss_outs = [self.model.batch_to_loss(data_batch, backward_step=False, extra_loss_func=self.extra_loss_func,
                     class_weights=class_weights, device=device) for data_batch in data_batches]
        outputs = torch.stack([(lo[1]/(len(loss_outs))) for lo in loss_outs])
        outputs = torch.sum(outputs, dim=0)
        loss = torch.stack([(lo[0]/(len(loss_outs))) for lo in loss_outs])
        loss = torch.sum(loss, dim=0)
        return loss, outputs

    def tta_predict(self, data_batches, device=None):
        preds = [self.model.predict(data_batch, device=device) for data_batch in data_batches]
        preds = torch.stack([(p/(len(preds))) for p in preds])
        preds = torch.sum(preds, dim=0)
        return preds

    def evaluate(self, dl, metric=None, pred_thresh=None, class_weights=None, do_tta=False, tta=None, num_tta=3, device=None):

        n_batches = num_batches(dl)
        if device is None:
            device = self.model.device
    
        if pred_thresh is None:
            pred_thresh = self.model.pred_thresh
        else:
            pred_thresh = pred_thresh

        if metric is None:
            metric = self.learn_metric
        else:
            metric = metric

        running_loss = 0.
        classifier = None

        if 'accuracy' in metric:
            try:
                class_names = self.dls.class_names
            except:
                class_names = self.model.class_names
            classifier = Classifier(class_names)

        y_pred = []
        y_prob = []
        y_true = []
        self.model.eval()
        rmse_ = 0.

        do_tta_ = None
        tta_ = None
        if hasattr(dl.dataset, 'do_tta'):
            do_tta_ = copy.deepcopy(dl.dataset.do_tta)
        dl.dataset.do_tta = do_tta

        if tta is not None:
            if not list_or_tuple(tta):
                tta = [tta]*num_tta
            if hasattr(dl.dataset, 'tta'):
                tta_ = copy.deepcopy(dl.dataset.tta)
            dl.dataset.tta = tta
        dl.dataset.num_tta = num_tta
        with torch.no_grad():
            for data_batch in dl:
                if list_or_tuple(data_batch):
                    loss, outputs = self.tta_batch_to_loss(data_batch, class_weights=class_weights, device=device)
                else:
                    loss, outputs = self.model.batch_to_loss(data_batch, backward_step=False,
                                                             extra_loss_func=self.extra_loss_func,
                                                             class_weights=class_weights, device=device)
                running_loss += loss
                if classifier is not None:
                    # labels = data_batch[1].to(device)
                    labels = self.model.open_batch(data_batch, device=device)['labels']
                    # labels = data_batch['label'].to(device)
                    if metric == 'accuracy':
                        classifier.update_accuracies(outputs, labels)
                        try:
                            y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                            prob, preds = torch.max(nn.Softmax()(outputs).cpu(), 1)
                            y_pred.extend(list(preds.cpu().numpy()))
                            y_prob.extend(list(prob.cpu().numpy()))
                        except:
                            pass
                    elif metric == 'multi_accuracy':
                        classifier.update_multi_accuracies(outputs, labels, pred_thresh)
                        try:
                            y_true += (list(labels.cpu().numpy()))
                            prob = nn.Sigmoid()(outputs).cpu()
                            y_prob += (list(prob.cpu().numpy()))
                        except:
                            pass
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.model.train()
        if tta_ is not None:
            dl.dataset.tta = tta_
        if do_tta_ is not None:
            dl.dataset.do_tta = do_tta_
        ret = {}
        # print('Running_loss: {:.5f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.5f}'.format(rmse_))
            ret['final_rmse'] = rmse_/n_batches

        ret['final_loss'] = running_loss/n_batches

        if classifier is not None:
            acc, class_acc = classifier.get_final_accuracies()
            ret['accuracy'] = acc
            if is_list(acc):
                ret['mean accuracy'] = torch.as_tensor(acc).mean().item()
            ret['class_accuracies'] = class_acc    
            # ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            try:
                ret['report'] = ClassificationReport(classification_report(y_true, y_pred,
                                                            target_names=class_names),
                                                            ret['accuracy'], ret['class_accuracies'])
                ret['confusion_matrix'] = ConfusionMatrix(confusion_matrix(y_true, y_pred), class_names)
            except:
                print('Classification report and confusion matrix not available.')
                ret['report'] = None
                ret['confusion_matrix'] = None
            try:
                yt = np.array(y_true)
                yp = np.array(y_prob)
                ret['roc_auc_score'] = roc_auc_score(yt, yp, multi_class='ovo', labels=class_names)
            except:
                print('ROC AUC score not available.')
                ret['roc_auc_score'] = None
        return ret

    def freeze(self):
        if self.model_splitter:
            freeze_params(params(self.model.model))
            p1,p2 = self.model_splitter(self.model.model)
            # freeze_params(p1)
            unfreeze_params(p2)
        
        else:
            self.model.freeze()
        self.is_frozen = True
    
    def unfreeze(self):
        self.model.unfreeze()
        self.is_frozen = False
        # unfreeze_params(self.model.parameters())


    def fine_tune(self, epochs=[12,30], frozen_lr=None, unfrozen_lr=None, metric='loss', save_every=None, save_best=True,
                  load_best_frozen=False, semi_sup=False, early_stopping_epochs=None, pred_thresh=None, class_weights=None,
                  print_every=3, validate_every=1, load_best_unfrozen=True, cycle_len=0, save_class=None, print_progress=True,
                  progressive_resizing=None, extra_loss_func=None, verbose=False, fit_scheduler=None, num_progress_classes=10):

        def frozen_fit(epochs, c_len, early_stopping_epochs=None):
            print(f'+{"-"*10}+')
            print(f'+  FROZEN  +')
            print(f'+{"-"*10}+')
            self.freeze()
            self.fit(epochs, lr=frozen_lr, metric=metric, load_best=load_best_frozen, semi_sup=semi_sup,
                    save_every=save_every, print_every=print_every, validate_every=validate_every,
                    cycle_len=c_len, save_class=save_class, print_progress=print_progress, save_best=save_best,
                    class_weights=class_weights, pred_thresh=pred_thresh, early_stopping_epochs=early_stopping_epochs,
                    extra_loss_func=extra_loss_func, verbose=verbose, fit_scheduler=fit_scheduler,
                    num_progress_classes=num_progress_classes)
        
        def unfrozen_fit(epochs, c_len, early_stopping_epochs):
            print()
            print(f'+{"-"*10}+')
            print(f'+ UNFROZEN +')
            print(f'+{"-"*10}+')
            self.unfreeze()
            self.fit(epochs, lr=unfrozen_lr, metric=metric, load_best=load_best_unfrozen, semi_sup=semi_sup,
                    save_every=save_every, print_every=print_every, validate_every=validate_every, save_best=save_best,
                    early_stopping_epochs=early_stopping_epochs, cycle_len=c_len, save_class=save_class,
                    print_progress=print_progress, pred_thresh=pred_thresh, class_weights=class_weights,
                    extra_loss_func=extra_loss_func, verbose=verbose, fit_scheduler=fit_scheduler,
                    num_progress_classes=num_progress_classes)

        if not list_or_tuple(epochs):
            epochs = [epochs, epochs]
        if not list_or_tuple(cycle_len):
            cycle_len = [cycle_len]*len(epochs)

        def run_fits():
            for i,e in enumerate(epochs):
                c_len = cycle_len[i]
                early_stopping = None
                if i==(len(epochs)-1):
                    early_stopping = early_stopping_epochs
                if i%2 == 0:
                    frozen_fit(epochs=e, c_len=c_len, early_stopping_epochs=early_stopping)
                else:
                    unfrozen_fit(epochs=e, c_len=c_len, early_stopping_epochs=early_stopping)

        if progressive_resizing is not None:
            for pr in progressive_resizing:
                if not list_or_tuple(pr):
                    pr = [pr]
                h,w = pr[0],pr[0]
                bs = None
                if len(pr) > 1:
                    bs = pr[1]
                if len(pr) > 2:
                    frozen_lr, unfrozen_lr = pr[2], pr[2]
                # set_lr(self.model.optimizer, p_lr)
                self.dls.progressive_resize(h=h, w=w, bs=bs)
                print(f'\nImage Shape: ({h},{w}), Batch Size: {self.dls.train.batch_size}\n')
                run_fits()
        else:
            run_fits()

    def find_lr(self, dl=None, init_value=1e-8, final_value=10., beta=0.98, class_weights=None, plot=False):

        print('\nFinding the ideal learning rate.')
        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.model.optimizer.state_dict())
        self.model.train()
        optimizer = self.model.optimizer
        if dl is None:
            dl = self.dls.train
        n_batches = num_batches(dl)
        num = n_batches-1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        set_lr(optimizer, lr)
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for data_batch in dl:
            batch_num += 1
            loss = self.model.batch_to_loss(data_batch, class_weights=class_weights, extra_loss_func=self.extra_loss_func)[0]
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) * loss
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.log_lrs, self.find_lr_losses = log_lrs,losses
                self.model.load_state_dict(model_state)
                self.model.optimizer.load_state_dict(optim_state)
                self.model.eval()
                if plot:
                    self.plot_find_lr()
                temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//7)]
                self.lr = (10**temp_lr)
                print('Found it: {}\n'.format(self.lr))
                return self.lr
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            #Update the lr for the next step
            lr *= mult
            set_lr(optimizer, lr)

        self.log_lrs, self.find_lr_losses = log_lrs,losses
        self.model.load_state_dict(model_state)
        self.model.optimizer.load_state_dict(optim_state)
        self.model.eval()
        if plot:
            self.plot_find_lr()
        temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//10)]
        self.lr = (10**temp_lr)
        print('Found it: {}\n'.format(self.lr))
        return self.lr
            
    def plot_find_lr(self):    
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate (log scale)")
        plt.plot(self.log_lrs,self.find_lr_losses)
        plt.show()

    def find_clr(self, dl=None, start_lr=1e-7, end_lr=1., lr_find_epochs=1, class_weights=None, plot=True):

        print('\nFinding the ideal max cyclic learning rate.')

        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.model.optimizer.state_dict())
        optimizer = self.model.optimizer
        set_lr(optimizer, start_lr)
        self.model.train()
        if dl is None:
            dl = self.dls.train
        n_batches = num_batches(dl)
        lr_lambda = lambda x: math.exp(x * math.log(end_lr/start_lr) / (lr_find_epochs*n_batches))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        lr_find_loss = []
        lr_find_lr = []
        iter = 0
        smoothing = 0.06
        best_loss = 100.
        last_lr = start_lr
        best_lr = 0.01
        for i in range(lr_find_epochs):
            if lr_find_epochs > 1:
                print("Epoch: {}".format(i+1))
            for data_batch in dl:
                loss = self.model.batch_to_loss(data_batch, class_weights=class_weights, extra_loss_func=self.extra_loss_func)[0]
                scheduler.step()
                if iter >= min(3,(n_batches//8)):
                    if loss <= best_loss:
                        best_loss = loss
                        best_lr = last_lr*0.1
                        # best_lr = last_lr
                    elif loss >= 4*best_loss:
                        self.model.load_state_dict(model_state)
                        self.model.optimizer.load_state_dict(optim_state)
                        self.model.eval()
                        if plot:
                            self.plot_find_clr(lr_find_lr, lr_find_loss)
                        print(f'Found it: {best_lr}\n')
                        return best_lr
                if iter!=0:
                    loss = smoothing  * loss + (1 - smoothing) * lr_find_loss[-1]
                    # print(loss, last_lr)
                lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
                lr_find_lr.append(lr_step)
                last_lr = lr_step
                lr_find_loss.append(loss)
                iter += 1
        
        self.model.load_state_dict(model_state)
        self.model.optimizer.load_state_dict(optim_state)
        self.model.eval()
        if plot:
            self.plot_find_clr(lr_find_lr, lr_find_loss)
        print(f'Found it: {best_lr}\n')
        return best_lr

    def plot_find_clr(self, lr_find_lr, lr_find_loss):
        plt.ylabel("loss")
        plt.xlabel("lr")
        plt.xscale("linear")
        plt.plot(lr_find_lr, lr_find_loss)
        plt.show()

    def __call__(self,name):
        for cb in self.cbs: getattr(cb,name,noop)()