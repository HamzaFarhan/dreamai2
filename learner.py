from .core import *

class Callback():
    
    def start_fit(self, obj): pass
    
    def start_epoch(self, obj): pass
    def start_training(self, obj): pass
    def on_train_loss(self, obj): pass
    def train_accum(self, obj): pass
    # def train_progress(self, obj): obj.print_train_progress({})
    def after_train_batch(self, obj): pass
    def after_train_batches(self, obj): pass
    def after_training(self, obj): pass
    
    def start_valid(self, obj): pass
    def start_valid_loop(self, obj): pass
    def valid_accum(self, obj): pass
    def after_valid_batch(self, obj): pass
    def update_valid_metrics(self, obj): pass
    def after_valid_loop(self, obj): pass
    # def valid_progress(self, obj): obj.print_valid_progress({})
    def after_valid(self, obj): pass
    
    def after_epoch(self, obj): pass
    def after_fit(self, obj): pass

class CheckpointCallback(Callback):
    def __init__(self, metric='loss', curr_best=None, save_every=1, save_name='model_checkpoint', load_best=True):

        self.load_best = load_best
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

    # def start_fit(self, obj):
        # self.curr_best = None

    def after_valid(self, obj):
        if (obj.curr_epoch+1) % self.save_every == 0:
            curr_metric = obj.val_ret[self.save_metric]
            if self.checker(curr_metric, self.curr_best, self.save_metric):
                top = f'\n**********Updating best {self.save_metric}**********\n'
                print(top)
                print(f'Previous best: {self.curr_best:.3f}')
                print(f'New best: {curr_metric:.3f}\n')
                bottom = '*'*(len(top)-1)
                print(f'{bottom}\n')
                self.curr_best = curr_metric
                torch.save({'model': obj.model.state_dict(),
                            'optimizer': obj.model.optimizer.state_dict(),
                            self.save_metric: self.curr_best
                            }, self.save_name)
    
    def after_fit(self, obj):
        if self.save_name.exists() and self.load_best:
            checkpoint = torch.load(self.save_name)
            load_state_dict(obj.model, checkpoint['model'])
            obj.model.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Best model loaded and model put in eval mode.')

class BasicCallback(Callback):
    def __init__(self): pass
    
    def start_training(self, obj):
        obj.model.train()
        obj.t0 = time.time()
        obj.t1 = time.time()
        obj.tr_running_loss = 0.
        obj.tr_batches = 0
#         setattr(obj, 'tr_batches', 0)
        
    def train_accum(self, obj):
        obj.tr_running_loss += obj.tr_batch_loss
    
    # def train_progress(self, obj):
    #     if obj.tr_batches % obj.fit_print_every == 0:
    #         elapsed = time.time()-obj.t1
    #         if elapsed > 60:
    #             elapsed /= 60.
    #             measure = 'min'
    #         else:
    #             measure = 'sec'
    #         batch_time = time.time()-obj.t0
    #         if batch_time > 60:
    #             batch_time /= 60.
    #             measure2 = 'min'
    #         else:
    #             measure2 = 'sec'
    #         obj.t0 = time.time()

    #         print('+----------------------------------------------------------------------+\n'
    #                 f"{time.asctime().split()[-2]}\n"
    #                 f"Time elapsed: {elapsed:.3f} {measure}\n"
    #                 f"Epoch:{obj.curr_epoch}/{obj.epochs-1}\n"
    #                 f"Batch: {obj.tr_batches}/{len(obj.tr_dl)}\n"
    #                 f"Batch training time: {batch_time:.3f} {measure2}\n"
    #                 f"Batch training loss: {obj.tr_batch_loss:.3f}\n"
    #                 f"Average training loss: {obj.tr_running_loss/(obj.tr_batches):.3f}\n"
    #               '+----------------------------------------------------------------------+\n'     
    #             )

    def after_train_batches(self, obj):
        obj.tr_ret = {}
        obj.tr_ret['loss'] = obj.tr_running_loss/len(obj.tr_dl)
        
    def start_valid(self, obj):
        obj.model.eval()
        obj.t2 = time.time()
        
    def start_valid_loop(self, obj):
        obj.val_running_loss = 0.
        obj.classifier = None
        if 'accuracy' in obj.fit_metric:
            obj.classifier = Classifier(obj.dls[1].class_names)

    def valid_accum(self, obj):
        obj.val_running_loss += obj.val_batch_loss
    
    def update_valid_metrics(self, obj):
        obj.val_ret = {}
        obj.val_ret['loss'] = obj.val_running_loss/len(obj.val_dl)
        if 'accuracy' in obj.fit_metric:
            obj.val_ret[obj.fit_metric], obj.val_ret['class_accuracies'] = obj.classifier.get_final_accuracies()
    
    # def valid_progress(self, obj):
    #     time_elapsed = time.time() - obj.t2
    #     if time_elapsed > 60:
    #         time_elapsed /= 60.
    #         measure = 'min'
    #     else:
    #         measure = 'sec'
        
    #     print('\n'+'/'*36+'\n'
    #             f"{time.asctime().split()[-2]}\n"
    #             f"Epoch {obj.curr_epoch}/{obj.epochs-1}\n"    
    #             f"Validation time: {time_elapsed:.6f} {measure}\n"    
    #             f"Epoch training loss: {obj.tr_ret['loss']:.6f}\n"                        
    #             f"Epoch validation loss: {obj.val_ret['loss']:.6f}")
    #     print('\\'*36+'\n')

    def after_valid(self, obj):
        obj.model.train()
    
    def after_fit(self, obj):
        torch.cuda.empty_cache()

def cbs_event(cbs, obj, event):
    for cb in cbs:
        getattr(cb, event)(obj)

class Learner():
    def __init__(self, model, dls, cbs=[BasicCallback()]):
        self.model = model
        self.dls = dls
        self.cbs = cbs
        assert len(cbs) > 0, print('Please pass some callbacks for training.')

    def print_train_progress(self, progress={}):
        if self.tr_batches % self.fit_print_every == 0:
            elapsed = time.time()-self.t1
            if elapsed > 60:
                elapsed /= 60.
                measure = 'min'
            else:
                measure = 'sec'
            batch_time = time.time()-self.t0
            if batch_time > 60:
                batch_time /= 60.
                measure2 = 'min'
            else:
                measure2 = 'sec'
            self.t0 = time.time()

            print('+----------------------------------------------------------------------+')
            print(f" {time.asctime().split()[-2]}")
            print(f" Time elapsed: {elapsed:.3f} {measure}")
            print(f" Epoch:{self.curr_epoch+1}/{self.epochs}")
            print(f" Batch: {self.tr_batches}/{len(self.tr_dl)}")
            print(f" Batch training time: {batch_time:.3f} {measure2}")
            print(f" Batch training loss: {self.tr_batch_loss:.3f}")
            print(f" Average training loss: {self.tr_running_loss/(self.tr_batches):.3f}")
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
        print(f"Epoch {self.curr_epoch+1}/{self.epochs}")
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

    def train_(self, dl, opt, print_every=3):
        
        self.tr_dl = dl
        self.tr_opt = opt
        self.tr_print_every = print_every
        
        cbs_event(self.cbs, self, 'start_training')
        
        for data_batch in dl:
            self.tr_batches += 1
            self.tr_batch_loss = self.model.batch_to_loss(data_batch)[0]
            
            cbs_event(self.cbs, self, 'on_train_loss')
            cbs_event(self.cbs, self, 'train_accum')
            self.print_train_progress()
            # cbs_event(self.cbs, self, 'train_progress')         
            cbs_event(self.cbs, self, 'after_train_batch')
            
        cbs_event(self.cbs, self, 'after_train_batches')
        return self.tr_ret
    
    def validate(self, dl):
        
        self.val_dl = dl
        cbs_event(self.cbs, self, 'start_valid')   
        
        self.validation_loop(dl)
        cbs_event(self.cbs, self, 'after_valid_loop')
        self.print_valid_progress()
        # cbs_event(self.cbs, self, 'valid_progress') 
        return self.val_ret
    
    def validation_loop(self, dl):
        
        cbs_event(self.cbs, self, 'start_valid_loop')
        
        for data_batch in dl:
                
            self.val_batch_loss, self.val_batch_outputs = self.model.val_batch_to_loss(data_batch,
                                                          metric=self.fit_metric, classifier=self.classifier)
            
            cbs_event(self.cbs, self, 'valid_accum')
            cbs_event(self.cbs, self, 'after_valid_batch')
            
        cbs_event(self.cbs, self, 'update_valid_metrics')
    
    def fit(self, epochs, metric='loss', print_every=3, validate_every=1):
        self.fit_epochs = epochs
        self.fit_metric = metric
        self.fit_print_every = print_every
        self.fit_validate_every = validate_every
        
        cbs_event(self.cbs, self, 'start_fit')

        for epoch in range(epochs):    
            self.curr_epoch, self.epochs = epoch, epochs
            cbs_event(self.cbs, self, 'start_epoch')
            self.tr_ret = self.train_(dl=self.dls[0], opt=self.model.optimizer, print_every=print_every)
            cbs_event(self.cbs, self, 'after_training')
            
            if validate_every and (epoch % validate_every == 0):            
                self.val_ret = self.validate(dl=self.dls[1])
                cbs_event(self.cbs, self, 'after_valid')
            cbs_event(self.cbs, self, 'after_epoch')
            
        cbs_event(self.cbs, self, 'after_fit')