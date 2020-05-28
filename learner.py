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
                torch.save({'model': self.model.state_dict(),
                            'optimizer': self.model.optimizer.state_dict(),
                            self.save_metric: self.curr_best
                            }, self.save_name)
    
    def after_fit(self):
        if self.save_name.exists() and self.load_best:
            checkpoint = torch.load(self.save_name)
            load_state_dict(self.model, checkpoint['model'])
            self.learner.model.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Best model loaded and model put in eval mode.')

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
        if 'accuracy' in self.fit_metric:
            self.learner.classifier = Classifier(self.dls.class_names)
    
    def after_val_batch(self):
        self.learner.val_running_loss += self.val_batch_loss
    
    def after_val_epoch(self):
        self.learner.val_ret = {}
        self.learner.val_ret['loss'] = self.val_running_loss/self.num_batches
        if 'accuracy' in self.fit_metric:
            self.learner.val_ret[self.fit_metric],\
            self.learner.val_ret['class_accuracies'] = self.classifier.get_final_accuracies()

    def after_valid(self):
        self.learner.model.train()

class Learner:
    def __init__(self, model, dls, model_splitter=None, cbs=[BasicCallback()]):

        store_attr(self, 'model,dls,model_splitter,cbs')
        for cb in cbs: cb.learner = self
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
        self.val_batch_loss = self.model.val_batch_to_loss(self.data_batch, metric=self.fit_metric,
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
    
    def fit(self, epochs, lr=None, metric='loss', print_every=3, validate_every=1, load_best=True):
        
        self.fit_epochs = epochs
        self.fit_metric = metric
        self.fit_print_every = print_every
        self.fit_validate_every = validate_every
        self.load_best = load_best
        
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