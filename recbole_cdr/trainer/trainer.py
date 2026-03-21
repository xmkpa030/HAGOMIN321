

r"""
recbole_cdr.trainer.trainer
################################
"""
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from time import time
from recbole.trainer import Trainer
from recbole.utils import set_color
from recbole_cdr.utils import train_mode2state
from recbole.utils import get_gpu_usage
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import NeighborSampler
from torch.utils.data import TensorDataset, DataLoader
from itertools import chain

class CrossDomainTrainer(Trainer):
    r"""Trainer for training cross-domain models. It contains four training mode: SOURCE, TARGET, BOTH, OVERLAP
    which can be set by the parameter of `train_epochs`
    """

    def __init__(self, config, model):
        super(CrossDomainTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, patience=8, verbose=True)
            if self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                else:
 
                    super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
            else:
                super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result



class HAGOTrainer(Trainer):

    def __init__(self, config, model):
        super(HAGOTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']
        self.save_step = self.config["save_step"]
        self.pretrain_method = self.config["pretrain_method"]
        self.reconstruct = self.config["reconstruct"]
        self.reaug_step = config['reaug_step']
        

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)
        
    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            "config": self.config,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "other_parameter": self.model.other_parameter(),
        }
        torch.save(state, saved_model_file)

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if self.config['reg_weight'] and weight_decay and weight_decay * self.config['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=8, verbose=True)
        return optimizer, scheduler
    
    def _pretrain_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
       
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        
        user_nodeset = TensorDataset(torch.arange(0, self.model.total_num_users, dtype=torch.int))
        item_nodeset = TensorDataset(torch.arange(self.model.total_num_users, self.model.total_num_users+self.model.total_num_items, dtype=torch.int))
        
        pretrain_loader_user = DataLoader(user_nodeset, batch_size=self.config['pretrain_batch_size'], shuffle=True)
        pretrain_loader_item = DataLoader(item_nodeset, batch_size=self.config['pretrain_batch_size'], shuffle=True)
        
        iter_data = (
            tqdm(
                chain(pretrain_loader_user, pretrain_loader_item),
                total=len(pretrain_loader_user)+len(pretrain_loader_item),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else chain(pretrain_loader_user, pretrain_loader_item)
        )
                
        count = 0
        for node_id in iter_data:
            count += 1
            self.optimizer.zero_grad()
            losses = loss_func([node_id, epoch_idx])
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return total_loss/count

    
    def pretrain(self, train_data, valid_data, verbose=True, show_progress=False):
        pretrained_path = os.path.join(
                    self.checkpoint_dir,
                    "{}-{}-{}-{}.pth".format(
                        self.config["model"], self.config["dataset"]["source_domain"], self.config["pretrain_method"], self.epochs
                    ),
                )
        if os.path.exists(pretrained_path):
            self.logger.info("Loading model: {}".format(pretrained_path))
            checkpoint = torch.load(pretrained_path)["state_dict"]
            for key in list(checkpoint.keys()):
                if 'target_encoder.' in key:
                    del checkpoint[key]
            self.model.load_state_dict(checkpoint, strict=False)

            return
            
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._pretrain_epoch(
                    train_data, epoch_idx, show_progress=show_progress
                )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()

            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            
            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    "{}-{}-{}-{}.pth".format(
                        self.config["model"], self.config["dataset"]["source_domain"], self.config["pretrain_method"], str(epoch_idx + 1)
                    ),
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = (
                    set_color("Saving current", "blue") + ": %s" % saved_model_file
                )
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result
    
    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)

            if self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    self.optimizer, self.scheduler = self._build_optimizer(params=filter(lambda p: p.requires_grad, list(self.model.parameters())), learning_rate=self.config['pretrain_learning_rate'])
                    self.pretrain(train_data, target_valid_data, verbose, show_progress)
                else:

                    para_list = list(self.model.gnn.parameters())
                    for i in range(len(para_list)):
                        para_list[i] = para_list[i].detach()
                        para_list[i].requires_grad = False
                    self.model.source_user_embedding.weight.requires_grad = False
                    self.model.source_item_embedding.weight.requires_grad = False
                    self.model.coord_embedding.weight.requires_grad = False
                        
                    self.optimizer, self.scheduler = self._build_optimizer(params=filter(lambda p: p.requires_grad, list(self.model.parameters())))
                    
                    super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
            else:
                self.optimizer, self.scheduler = self._build_optimizer(params=filter(lambda p: p.requires_grad, list(self.model.parameters())))
                super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
                
        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result
    
