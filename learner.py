from .core import *
from .data import *

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

class CheckpointCallback(Callback):
    def __init__(self, metric='loss', curr_best=None, save_every=1, save_name='model_checkpoint'):

        self.save_metric = metric
        self.save_every = save_every
        if curr_best is None:
            if metric == 'loss':
                self.curr_best = 100.
            else:
                self.curr_best = 0.
        else:
            self.curr_best = curr_best
        if save_name [-4:] != '.pth' or save_name [-4:] != '.pkl':
            save_name = save_name+'.pth'
        checkpoint_folder = 'dai_model_checkpoints'
        os.makedirs(checkpoint_folder, exist_ok=True)
        self.save_name = Path(checkpoint_folder)/save_name

    def checker(self, curr, best, metric='loss'):
        if metric == 'loss':
            return curr <= best
        return curr >= best

    # def before_fit(self):
        # self.curr_best = None

    def after_valid(self):
        if (self.curr_epoch+1) % self.save_every == 0:
            curr_metric = self.val_ret[self.save_metric]
            if self.checker(curr_metric, self.curr_best, self.save_metric):
                top = f'\n**********Updating best {self.save_metric}**********\n'
                print(top)
                print(f'Previous best: {self.curr_best:.3f}')
                print(f'New best: {curr_metric:.3f}\n')
                bottom = '*'*(len(top)-2)
                print(f'{bottom}\n')
                self.curr_best = curr_metric
                checkpoint = self.model.checkpoint_dict()
                checkpoint[self.save_metric] = self.curr_best
                if 'accuracy' in self.save_metric:
                    checkpoint['class_names'] = self.dls.class_names
                torch.save(checkpoint, self.save_name)
                # torch.save({'model': self.model.state_dict(),
                #             'optimizer': self.model.optimizer.state_dict(),
                #             self.save_metric: self.curr_best
                #             }, self.save_name)
    
    def after_fit(self):
        if self.save_name.exists() and self.load_best:
            checkpoint = torch.load(self.save_name)
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
    
    def after_val_batch(self):
        self.learner.val_running_loss += self.val_batch_loss
    
    def after_val_epoch(self):
        self.learner.val_ret = {}
        self.learner.val_ret['loss'] = self.val_running_loss/self.num_batches
        if 'accuracy' in self.learn_metric:
            self.learner.val_ret[self.learn_metric],\
            self.learner.val_ret['class_accuracies'] = self.classifier.get_final_accuracies()

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

        if self.learn_metric == 'multi_accuracy':
            p = nn.Sigmoid()(self.pred_out).cpu()[0]
            bools = (p >= self.pred_thresh)
            print(bools)
            c = np.array(self.dls.class_names)[bools]
            self.learner.pred_out = {'probs': p, 'bools':bools, 'pred': c}
        # if 'accuracy' in self.learn_metric:


class Learner:
    def __init__(self, model, dls, model_splitter=None, metric='loss', cbs=[BasicCallback()]):

        store_attr(self, 'model,dls,model_splitter,cbs')
        self.learn_metric = metric
        for cb in cbs: cb.learner = self
        self.model.normalize = dls.normalize
        self.model.denorm = dls.denorm
        self.model.img_mean = dls.img_mean
        self.model.img_std = dls.img_std
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
                total_time = (batch_time*self.num_batches*self.fit_epochs) + (self.fit_epochs/self.fit_validate_every)
                total_time += total_time/5
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
            print(f" Time elapsed: {elapsed:.3f}{measure} / {total_time:.3f}{total_measure}")
            print(f" Epoch:{self.curr_epoch+1}/{self.fit_epochs}")
            print(f" Batch: {self.batch_num+1}/{self.num_batches}")
            print(f" Batch training time: {batch_time:.3f} {measure2}")
            print(f" Batch training loss: {self.tr_batch_loss:.3f}")
            print(f" Average training loss: {self.tr_running_loss/(self.batch_num+1):.3f}")
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
                    if len(v) <= 5:
                        print(f"Epoch validation {k}:")
                        for x in v:
                            print(f'    {x}')
                else:
                    print(f"Epoch validation {k}: {v:.6}")

        print('\\'*36+'\n')
    
    def train_batch(self):
        self('before_train_batch')
        self.tr_batch_loss = self.model.batch_to_loss(self.data_batch)[0]
        self('after_train_batch')
    
    def val_batch(self):
        self('before_val_batch')
        self.val_batch_loss = self.model.val_batch_to_loss(self.data_batch, metric=self.learn_metric,
                                                           classifier=self.classifier)[0]
        self('after_val_batch')
    
    def train_epoch(self):
        self('before_train_epoch')
        dl = self.dls.train
        self.num_batches = len(dl)
        for self.batch_num, self.data_batch in enumerate(dl):
            self.train_batch()
            self.print_train_progress()
        self('after_train_epoch')
    
    def val_epoch(self):
        self('before_val_epoch')
        dl = self.dls.valid
        self.num_batches = len(dl)
        with torch.no_grad():
            for self.batch_num, self.data_batch in enumerate(dl):
                self.val_batch()
        self('after_val_epoch')
        self.print_valid_progress()
    
    def fit(self, epochs, lr=None, metric='loss', print_every=3, validate_every=1, load_best=True, self_sup=False):
        
        self.fit_epochs = epochs
        self.learn_metric = metric
        self.fit_print_every = print_every
        self.fit_validate_every = validate_every
        self.load_best = load_best
        self.self_sup = self_sup
        
        if lr:
            set_lr(self.model.optimizer, lr)

        self('before_fit')
        for self.curr_epoch in range(epochs):
            
            self('before_training')
            self.train_epoch()
            self('after_training')
            
            if validate_every and (self.curr_epoch % validate_every == 0):
                self('before_valid')
                self.val_epoch()
                self('after_valid')                
        self('after_fit')

    def predict(self, x, pred_thresh=None, device=None):

        if pred_thresh is None:
            self.pred_thresh = self.model.pred_thresh
        else:
            self.pred_thresh = pred_thresh

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
        dl = self.dls.test
        dset = dl.dataset
        tfms = dset.tfms
        # tfms = None
        self.pred_set = PredDataset(x, tfms=tfms)
        # pred_dl = DataLoader(self.pred_set, batch_size=bs)
        self.pred_outs = []
        for idx,data_batch in enumerate(self.pred_set):
            self.pred_out = self.model.predict(data_batch, device=device)
            self('after_predict')
            self.pred_outs.append(self.pred_out)

        return self.pred_outs

    def evaluate(self, dl, metric=None, pred_thresh=None, device=None):

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
        with torch.no_grad():
            for data_batch in dl:
                loss, outputs = self.model.batch_to_loss(data_batch, backward_step=False, device=device)
                running_loss += loss
                if classifier is not None:
                    labels = data_batch[1].to(device)
                    if metric == 'accuracy':
                        classifier.update_accuracies(outputs, labels)
                        try:
                            y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                            prob, preds = torch.max(nn.Softmax()(outputs).cpu(), 1)
                            # m = torch.max(p, 1)
                            # prob, preds = torch.max(torch.exp(outputs), 1)
                            y_pred.extend(list(preds.cpu().numpy()))
                            y_prob.extend(list(prob.cpu().numpy()))
                        except:
                            pass
                    elif metric == 'multi_accuracy':
                        classifier.update_multi_accuracies(outputs, labels, pred_thresh)
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.model.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dl)

        ret['final_loss'] = running_loss/len(dl)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            try:
                ret['report'] = ClassificationReport(classification_report(y_true, y_pred, target_names=class_names),
                                                     ret['accuracy'], ret['class_accuracies'])
                ret['confusion_matrix'] = ConfusionMatrix(confusion_matrix(y_true, y_pred), class_names)
                try:
                    ret['roc_auc_score'] = roc_auc_score(np.array(y_true), np.array(y_prob),
                                           multi_class='ovo', labels=class_names)
                except:
                    pass
            except:
                print('Classification report and confusion matrix not available.')
                ret['report'] = None
                ret['confusion_matrix'] = None
        return ret

    def freeze(self):
        if self.model_splitter:
            freeze_params(params(self.model))
            p1,p2 = self.model_splitter(self.model)
            # freeze_params(p1)
            unfreeze_params(p2)
        
        else:
            self.model.freeze()
    
    def unfreeze(self):
        self.model.unfreeze()
        # unfreeze_params(self.model.parameters())


    def fine_tune(self, frozen_epochs=2, unfrozen_epochs=5, frozen_lr=0.001, unfrozen_lr=0.001,
                  metric='loss', load_best_frozen=False):

        print(f'+{"-"*10}+')
        print(f'+  FROZEN  +')
        print(f'+{"-"*10}+')
        self.freeze()
        self.fit(frozen_epochs, lr=frozen_lr, metric=metric, load_best=load_best_frozen)
        print()
        print(f'+{"-"*10}+')
        print(f'+ UNFROZEN +')
        print(f'+{"-"*10}+')
        self.unfreeze()
        self.fit(unfrozen_epochs, lr=unfrozen_lr, metric=metric, load_best=True)

    def find_lr(self, dl=None, init_value=1e-8, final_value=10., beta=0.98, plot=False):

        print('\nFinding the ideal learning rate.')

        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.model.optimizer.state_dict())
        optimizer = self.model.optimizer
        if dl is None:
            dl = self.dls.train
        num = len(dl)-1
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
            loss = self.model.batch_to_loss(data_batch)[0]
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) * loss
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.log_lrs, self.find_lr_losses = log_lrs,losses
                self.model.load_state_dict(model_state)
                self.model.optimizer.load_state_dict(optim_state)
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

    def __call__(self,name):
        for cb in self.cbs: getattr(cb,name,noop)()