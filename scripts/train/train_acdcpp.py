# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import copy
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from composer import Trainer
from composer.core import Evaluator
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from transformers import PreTrainedTokenizerBase
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig


from llmfoundry import (COMPOSER_MODEL_REGISTRY, ComposerHFCausalLM,
                        MPTForCausalLM, build_finetuning_dataloader,
                        build_text_denoising_dataloader)
from llmfoundry.data.text_data import build_text_dataloader
from llmfoundry.utils.builders import (build_algorithm, build_callback,
                                       build_icl_evaluators, build_logger,
                                       build_optimizer, build_scheduler,
                                       build_tokenizer)
from llmfoundry.utils.config_utils import (log_config, pop_config,
                                           process_init_device,
                                           update_batch_size_info)

from hf_bert import create_hf_bert_mlm


# ============== ACDC implementation as Composer algorithm ====================
from composer.core import Algorithm, Event, State
import re
import torch
import wandb

class ACDC(Algorithm):
    @torch.no_grad()
    def __init__(self, model, params_re: str, sparsity_structure: str, is_global: bool, schedule: List[dict], pruner: str):        
        self.sparsity_structure = sparsity_structure
        self.is_global = is_global
        self.pruner = pruner
        self.first_time_here = True

        self.step = 0
        self.schedule = self._process_schedule(schedule)
        self.current_sparsity = None # None means dense

        # self.sparse_schedule = [False] * self.warmup
        # assert (self.total_num_steps - self.warmup - self.cooldown) / self.acdc_freq == (self.total_num_steps - self.warmup - self.cooldown) // self.acdc_freq, 'ACDC frequency must divide total number of steps'
        # assert (self.total_num_steps - self.warmup - self.cooldown) // self.acdc_freq % 2 != 0, f'ACDC frequency must be odd, got self.total_num_steps={self.total_num_steps}'
        # for i in range(0, (self.total_num_steps - self.warmup - self.cooldown) // self.acdc_freq):
        #     self.sparse_schedule += [True] * self.acdc_freq if i % 2 == 0 else [False] * self.acdc_freq
        # self.sparse_schedule += [True] * self.cooldown
        # assert len(self.sparse_schedule) == self.total_num_steps, "Sparse schedule length doesn't match total number of steps"

        # collect params for pruning
        self.prune_params = []
        self.prune_params_names = []
        for name, param in model.named_parameters():
            if re.match(params_re, name):  # skip the r, needed for yaml
                self.prune_params.append(param)
                self.prune_params_names.append(name)

        print(f'[ACDC] Pruning {self.prune_params_names} parameters')

    def _process_schedule(self, schedule):
        clean_schedule = []
        current_step = 0
        for i, item in enumerate(schedule):
            if 'end' not in item:
                assert i == len(schedule) - 1, 'Only the last item in the schedule can have no end'
                item['end'] = int(1e9)
            
            assert current_step < item['end'], 'Schedule items must be in increasing order'

            sp_parts = str(item['sparsity']).split('/')
            if len(sp_parts) == 1: # it's just a single number
                sparsity = float(sp_parts[0])
                clean_schedule.append({
                    'sparsity': sparsity,
                    'end': item['end'],
                    'reset_optimizer': item.get('reset_optimizer', False),
                })
            elif len(sp_parts) == 2: # it's an alternation
                assert 'freq' in item, f'Frequency not specified for {item["sparsity"]}'

                sparsities = [float(sp) for sp in sp_parts]

                assert (item['end'] - current_step) % item['freq'] == 0, f'Frequency {item["freq"]} does not divide the number of steps {item["end"] - current_step}'
                for i in range((item['end'] - current_step) // item['freq']):
                    clean_schedule.append({
                        'sparsity': sparsities[i % 2],
                        'end': current_step + (i + 1) * item['freq'],
                        'reset_optimizer': item.get('reset_optimizer', False)
                    })
            else: # it's gradual acdc
                assert len(sp_parts) == 3, f'Invalid sparsity {item["sparsity"]}'
                typ = sp_parts[0]
                assert typ in ['linear'], f'Unknown gradual type {typ}'

                low, high = float(sp_parts[1]), float(sp_parts[2])
                assert low < high, f'Invalid sparsity {item["sparsity"]}'

                assert (item['end'] - current_step) % item['freq'] == 0, f'Frequency {item["freq"]} does not divide the number of steps {item["end"] - current_step}'
                num_phases = (item['end'] - current_step) // item['freq']
                for i in range(num_phases):
                    sparsity = low if i % 2 == 0 else ((high - low) * (i + 1) / num_phases + low)
                    clean_schedule.append({
                        'sparsity': sparsity,
                        'end': current_step + (i + 1) * item['freq'],
                        'reset_optimizer': item.get('reset_optimizer', False)
                    })

            current_step = item['end']
            
        
        print(f'[ACDC] Schedule: {clean_schedule}')
        return clean_schedule

    def _get_current_agenda(self):
        sparsity = None
        time_to_reset_optim = False
        for item in self.schedule:
            if self.step < item['end']:
                sparsity = item['sparsity']
                reset_optim = item['reset_optimizer']
                return {'sparsity': sparsity, 'reset_optim': reset_optim and time_to_reset_optim}
            elif self.step == item['end']:
                time_to_reset_optim = True
        assert False, f'No agenda found for step {self.step}'

    @torch.no_grad()
    def initialize_masks(self, model):
        self.current_sparsity = None
        
        # initialize masks
        self.masks = []
        for param in self.prune_params:
            self.masks.append(torch.ones_like(param, requires_grad=False))

        # for logging
        self.num_params = sum([param.numel() for param in model.parameters()])
        self.num_prunable = sum([param.numel() for param in self.prune_params])

    @torch.no_grad()
    def reset_optimizers(self, optimizers):
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'momentum_buffer' in state: # for SGD
                        state['momentum_buffer'].fill_(0.)
                    if 'exp_avg' in state and 'exp_avg_sq' in state and 'step' in state: # for Adam
                        state['exp_avg'].fill_(0.)
                        state['exp_avg_sq'].fill_(0.)
                        state['step'] = torch.zeros_like(state['step'])
        print('[ACDC] Optimizers were reset')

    @torch.no_grad()
    def apply_mask(self):
        if self.current_sparsity is None:
            return
        
        for param, mask in zip(self.prune_params, self.masks, strict=True):
            param.data *= mask
            # if hasattr(param, 'grad'):
            #     param.grad *= mask
            
    @torch.no_grad()
    def prune_magnitude_unstr(self, sparsity):
        assert self.pruner == 'magnitude'
        assert isinstance(sparsity, float) and 0 < sparsity < 1, f'Invalid sparsity {sparsity}'

        scores = [torch.abs(param.data).reshape(-1) for param in self.prune_params]
        if self.is_global:
            scores = [torch.cat(scores)]

        thresholds = []
        pick_thresh_probs = [] # useful when there are ties, especially when threshold is 0
        for score in scores:
            num_zeros = int(score.numel() * sparsity)
            threshold = torch.kthvalue(score, num_zeros)[0]
            thresholds.append(threshold)
            num_eq_thresh = (score == threshold).sum().item()
            num_le_thresh = (score <= threshold).sum().item()
            if num_zeros < num_le_thresh and num_eq_thresh >= num_le_thresh - num_zeros:
                pick_thresh_probs.append((num_le_thresh - num_zeros) / num_eq_thresh)
            else:
                pick_thresh_probs.append(0.)

        if self.is_global:
            thresholds = thresholds * len(self.prune_params)
            pick_thresh_probs = pick_thresh_probs * len(self.prune_params)

        # if sparsity > 0 and sparsity < 0.7:
        #     if self.gpu == 0:
        #         breakpoint()
        #     torch.distributed.barrier()

        for param, mask, threshold, ptp in zip(self.prune_params, self.masks, thresholds, pick_thresh_probs, strict=True):
            mask.data = torch.where(torch.abs(param.data) > threshold, torch.ones_like(mask), torch.zeros_like(mask))

            if ptp > 0:
                eq_mask = torch.where(
                    torch.logical_and(torch.abs(param.data) == threshold, torch.bernoulli(torch.ones_like(mask) * ptp)),
                    torch.ones_like(mask),
                    torch.zeros_like(mask),
                )
                mask.data = torch.logical_or(mask.data, eq_mask)
            
            print(f'[ACDC] A mask was generated with sparsity {1 - mask.sum().item() / mask.numel()}.')
    
    @torch.no_grad()
    def prune_magnitude_vnm(self, v, n, m, shuffle_cols=True):
        assert self.pruner == 'magnitude'
        assert not self.is_global, 'global pruning not supported for vnm'
        assert n == 2, 'only n=2 is supported for vnm'

        for param, mask in zip(self.prune_params, self.masks, strict=True):
            p, q = param.shape
            num_blocks = p // v * q // m
            score = param.reshape(p // v, v, q // m, m).permute(0, 2, 1, 3).reshape(num_blocks, v, m)
            
            if shuffle_cols: # shuffle the columns of each block to give zero columns equal chances of being picked
                rand_perm = torch.argsort(torch.rand(num_blocks, 1, m, device=score.device), dim=-1).repeat(1, v, 1)
                score = torch.gather(score, -1, rand_perm)

            combs = torch.combinations(torch.arange(m), r=2*n).to(score.device)

            # TODO: optimize this. maybe break combs into chunks and do the following in sequence to save memory
            combed_score = score.permute(0, 2, 1)[:, combs, :].permute(0, 1, 3, 2).reshape(num_blocks, -1, v, 2*n)

            topk_val, topk_idx = torch.topk(combed_score, k=n, dim=-1)

            sumns = topk_val.sum(-1).sum(-1)

            top_comb_idx = torch.argmax(sumns, dim=-1)

            selected_comb = combs[top_comb_idx]
            top_comb_nm_idx = topk_idx[torch.arange(num_blocks, device=score.device), top_comb_idx]

            merged_idx = torch.gather(
                selected_comb.unsqueeze(1).repeat(1, v, 1),
                -1,
                top_comb_nm_idx
            )

            new_mask = torch.zeros(num_blocks, v, m, dtype=torch.bool, device=score.device)
            new_mask.scatter_(-1, merged_idx, 1)

            if shuffle_cols:
                new_mask = torch.gather(new_mask, -1, torch.argsort(rand_perm, dim=-1))

            new_mask = new_mask.reshape(p // v, q // m, v, m).permute(0, 2, 1, 3).reshape(p, q)
            mask.data = new_mask

            print(f'[ACDC] A venom mask was generated with sparsity {1 - mask.sum().item() / mask.numel()}.')

    @torch.no_grad()
    def prune_magnitude_str(self, sparsity, do_log=True):
        assert self.pruner == 'magnitude'
        assert isinstance(sparsity, float) and 0 < sparsity < 1, f'Invalid sparsity {sparsity}'
        
        row_scores = [torch.abs(param.data).sum(-1) for param in self.prune_params]

        if self.is_global:
            row_scores = [torch.cat(row_scores)]

        thresholds = []
        pick_thresh_probs = [] # useful when there are ties, especially when threshold is 0
        for row_score in row_scores:
            num_zeros = int(row_score.numel() * sparsity)
            threshold = torch.kthvalue(row_score, num_zeros)[0]
            thresholds.append(threshold)
            num_eq_thresh = (row_score == threshold).sum().item()
            num_le_thresh = (row_score <= threshold).sum().item()
            if num_zeros < num_le_thresh and num_eq_thresh >= num_le_thresh - num_zeros:
                pick_thresh_probs.append((num_le_thresh - num_zeros) / num_eq_thresh)
            else:
                pick_thresh_probs.append(0.)

        if self.is_global:
            thresholds = thresholds * len(self.prune_params)
            pick_thresh_probs = pick_thresh_probs * len(self.prune_params)

        # if sparsity > 0 and sparsity < 0.7:
        #     if self.gpu == 0:
        #         breakpoint()
        #     torch.distributed.barrier()

        for param, mask, threshold, ptp in zip(self.prune_params, self.masks, thresholds, pick_thresh_probs, strict=True):
            row_abs_sums = torch.abs(param.data).sum(-1)
            row_mask = torch.where(row_abs_sums > threshold, torch.ones_like(row_abs_sums, dtype=mask.dtype), torch.zeros_like(row_abs_sums, dtype=mask.dtype))

            if ptp > 0:
                eq_mask = torch.where(
                    torch.logical_and(row_abs_sums == threshold, torch.bernoulli(torch.ones_like(row_mask) * ptp)),
                    torch.ones_like(row_mask),
                    torch.zeros_like(row_mask),
                )
                row_mask = torch.logical_or(row_mask, eq_mask)
            
            mask.data = row_mask.unsqueeze(-1).repeat(1, mask.data.shape[-1])

            if do_log:
                print(f'[ACDC] A mask was generated with sparsity {1 - mask.sum().item() / mask.numel()}.')

    @torch.no_grad()
    def prune_magnitude_cas(self, sparsity, col_transfer_ratio=0.9):
        # first let's prune the rows
        self.prune_magnitude_str(sparsity, do_log=False)

        mapping = {
            'self_attn.q_proj': [('prev', 'mlp.down_proj')],
            'self_attn.k_proj': [('prev', 'mlp.down_proj')],
            'self_attn.v_proj': [('prev', 'mlp.down_proj')],
            'mlp.up_proj': [('current', 'self_attn.o_proj')],
            'mlp.gate_proj': [('current', 'self_attn.o_proj')],
            'mlp.down_proj': [('current', 'mlp.up_proj'), ('current', 'mlp.gate_proj')]
        }

        # let's find the mask of the non-pruned rows first
        pruned_row_masks = {}
        for name, mask in zip(self.prune_params_names, self.masks, strict=True):
            pruned_row_masks[name] = mask.data.sum(-1) != 0

        # now let's prune the columns
        for name, mask, param in zip(self.prune_params_names, self.masks, self.prune_params, strict=True):
            if '.weight' not in name:
                continue
            
            for key, value in mapping.items():
                if key in name:
                    layer = int(name.split('layers.')[-1].split('.self_attn')[0].split('.mlp')[0])
                    col_mask_and = mask.new_ones(mask.shape[-1])
                    for src_layer, src_key in value:
                        src_name = name.replace(key, src_key)
                        if src_layer == 'prev':
                            if layer == 0: # no prev
                                continue
                            src_name = src_name.replace(f'.{layer}.', f'.{layer - 1}.')
                        col_mask_and = torch.logical_and(col_mask_and, pruned_row_masks[src_name])

                    if torch.all(col_mask_and):
                        continue
                    
                    col_scores = torch.abs(param.data).sum(0)[torch.logical_not(col_mask_and)]
                    num_zeros = int(col_scores.numel() * col_transfer_ratio)
                    threshold = torch.kthvalue(col_scores, num_zeros)[0]

                    ptp = 0 # to supprot acdc++, where we decrease sprasity to somewhere more than 0
                    num_eq_thresh = (col_scores == threshold).sum().item()
                    num_le_thresh = (col_scores <= threshold).sum().item()
                    if num_zeros < num_le_thresh and num_eq_thresh >= num_le_thresh - num_zeros:
                        ptp = (num_le_thresh - num_zeros) / num_eq_thresh

                    col_sub_mask = torch.where(col_scores > threshold, torch.ones_like(col_scores, dtype=mask.dtype), torch.zeros_like(col_scores, dtype=mask.dtype))

                    if ptp > 0:
                        eq_mask = torch.where(
                            torch.logical_and(col_scores == threshold, torch.bernoulli(torch.ones_like(col_sub_mask) * ptp)),
                            torch.ones_like(col_sub_mask),
                            torch.zeros_like(col_sub_mask),
                        )
                        col_sub_mask = torch.logical_or(col_sub_mask, eq_mask)

                    mask.data[:, torch.logical_not(col_mask_and)] *= col_sub_mask.unsqueeze(0).repeat(mask.shape[0], 1)

        # print(f'[ACDC CAS] {name}')
        # print(f'unstr={1 - mask.sum().item() / mask.numel():.3f}, row={(mask.sum(-1) == 0).float().mean():.3f}, col={(mask.sum(0) == 0).float().mean():.3f}')
        # print('-' * 80)
                


    # @torch.no_grad()
    # def prune_fisher_blocksize1(self, optimizers):
    #     optim = optimizers[0]

    #     scores = [(param.data**2 * optim.state[param]['exp_avg_sq']).reshape(-1) for param in self.prune_params]
    #     if self.is_global:
    #         scores = [torch.cat(scores)]

    #     thresholds = []
    #     for score in scores:
    #         threshold = torch.kthvalue(score, int(score.numel() * self.sparsity))[0]
    #         thresholds.append(threshold)

    #     if self.is_global:
    #         thresholds = thresholds * len(self.prune_params)
    #     for param, mask, threshold in zip(self.prune_params, self.masks, thresholds, strict=True):
    #         mask.data = torch.where((param.data**2 * optim.state[param]['exp_avg_sq']) < threshold, torch.zeros_like(mask), torch.ones_like(mask))


    def get_sparsity_logs(self):
        to_log = {}
        num_zeros_params = 0
        num_zeros_masks = 0
        num_all_zero_rows = 0
        num_all_zero_cols = 0
        num_all_rows = 0
        num_all_cols = 0
        for name, mask, param in zip(self.prune_params_names, self.masks, self.prune_params):
            num_pruned = (mask.data == 0).sum().item()
            num_pruned_rows = (mask.sum(-1) == 0).sum().item()
            num_pruned_cols = (mask.sum(0) == 0).sum().item()
            to_log['acdc_perparam/' + name] = num_pruned / mask.numel()
            to_log['acdc_perparam_row/' + name] = num_pruned_rows / mask.shape[0]
            to_log['acdc_perparam_col/' + name] = num_pruned_cols / mask.shape[1]
            num_zeros_masks += num_pruned
            num_zeros_params += (param.data == 0).sum().item()
            num_all_zero_rows += num_pruned_rows
            num_all_zero_cols += num_pruned_cols
            num_all_rows += mask.shape[0]
            num_all_cols += mask.shape[1]

        to_log['acdc/prunable_sparsity_masks'] = num_zeros_masks / self.num_prunable
        to_log['acdc/prunable_sparsity_params'] = num_zeros_params / self.num_prunable
        to_log['acdc/prunable_row_sparsity'] = num_all_zero_rows / num_all_rows
        to_log['acdc/prunable_col_sparsity'] = num_all_zero_cols / num_all_cols
        to_log['acdc/num_params'] = self.num_params
        to_log['acdc/num_prunable'] = self.num_prunable
        return to_log

    def match(self, event, state):
        # BATCH_START = prune/unprune when needed
        # BATCH_END = apply mask when sparse phase
        return event == Event.BATCH_START or event == Event.BATCH_END

    @torch.no_grad()
    def prune(self, sparsity):
        if sparsity is None or sparsity <= 0:
            for mask in self.masks:
                mask.fill_(1.)
            print('masks were reset to all ones')
            return

        if self.pruner == 'magnitude':
            if self.sparsity_structure == 'unstr':
                self.prune_magnitude_unstr(sparsity)
            elif self.sparsity_structure == 'str':
                self.prune_magnitude_str(sparsity)
            elif self.sparsity_structure == 'vnm':
                v, n, m = sparsity.split(':')
                self.prune_magnitude_vnm(int(v), int(n), int(m))
            elif self.sparsity_structure == 'cas':
                self.prune_magnitude_cas(sparsity)
        else:
            raise NotImplementedError(f"Pruner {self.pruner} not implemented")    
        

    @torch.no_grad()
    def apply(self, event, state, logger):
        if self.first_time_here:  # late initialization because of late init of torch.distributed
            self.initialize_masks(state.model)
            self.gpu = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
            self.first_time_here = False

        if event == Event.BATCH_START:
            self.step += 1

            assert self.pruner == 'magnitude', 'Only magnitude pruning is supported for now'
            agenda = self._get_current_agenda()
            if agenda['sparsity'] != self.current_sparsity:
                self.prune(agenda['sparsity'])
                if agenda['reset_optim']:
                    self.reset_optimizers(state.optimizers)
                self.current_sparsity = agenda['sparsity']
            
            self.apply_mask()

        elif event == Event.BATCH_END:  # just apply masks and log
            self.apply_mask()
            if self.gpu == 0:
                wandb.log(self.get_sparsity_logs())
# ===============================================================================

def validate_config(cfg: DictConfig):
    """Validates compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if 'eval_loader' in cfg:
        loaders.append(cfg.eval_loader)
    for loader in loaders:
        if loader.name == 'text':
            if cfg.model.name in ['hf_prefix_lm', 'hf_t5']:
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text " ' +\
                    f'dataloader. Please use the "text_denoising" dataloader to pre-train that model type.')
        elif loader.name == 'text_denoising':
            if cfg.model.name == 'hf_causal_lm':
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text_denoising" ' +\
                    f'dataloader. Please use the "text" dataloader to pre-train that model type.')
            if loader.mixture_of_denoisers.decoder_only_format and cfg.model.name == 'hf_t5':
                warnings.warn(
                    'Model type "hf_t5" requires `decoder_only_format` to be ``False``. ' +\
                    'Overriding `decoder_only_format` from ``True`` to ``False``.')
                loader.mixture_of_denoisers.decoder_only_format = False
            if (not loader.mixture_of_denoisers.decoder_only_format
               ) and cfg.model.name == 'hf_prefix_lm':
                warnings.warn(
                    'Model type "hf_prefix_lm" requires `decoder_only_format` to be ``True``. ' +\
                    'Overriding `decoder_only_format` from ``False`` to ``True``.')
                loader.mixture_of_denoisers.decoder_only_format = True

    if 'icl_tasks' in cfg:
        if cfg.model.name == 'hf_t5':
            raise ValueError(
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".'
            )

    if (cfg.model.get('fc_type', 'torch') != 'te' and 'te' not in cfg.model.get(
            'ffn_config', {}).get('ffn_type', 'mptmlp') and
            'fp8' in cfg.precision):
        warnings.warn(
            "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
            +
            "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision."
        )

    if (cfg.model.get('fc_type', 'torch') == 'te' or
            'te' in cfg.model.get('ffn_config', {}).get('ffn_type', 'mptmlp')):
        fsdp_config = cfg.get('fsdp_config', None)
        act_ckpt = fsdp_config.get('activation_checkpointing', False)
        act_ckpt_reentrant = fsdp_config.get(
            'activation_checkpointing_reentrant', True)
        if fsdp_config is not None and act_ckpt == True and act_ckpt_reentrant == False:
            warnings.warn(
                '`te.Linear` layers do not support activation_checkpointing with '
                + '`activation_checkpointing_reentrant = False`. ' +
                'Setting cfg.fsdp_config.activation_checkpointing_reentrant=True.'
            )
            cfg.fsdp_config.activation_checkpointing_reentrant = True

    if 'te' in cfg.model.get('ffn_config', {}).get('ffn_type', 'mptmlp'):
        warnings.warn(
            '`te.LayerNormMLP` requires has issues with torch._dynamo. ' +
            'Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager.'
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore


def build_composer_model(model_cfg: DictConfig,
                         tokenizer: PreTrainedTokenizerBase):
    if model_cfg.name == 'hf_bert':
        return create_hf_bert_mlm(
            pretrained_model_name_or_path=model_cfg.pretrained_model_name_or_path,
            pretrained=model_cfg.get('pretrained', None),
            model_config=model_cfg.get('model_config', None),
            tokenizer_name=model_cfg.get('tokenizer_name', None),
            gradient_checkpointing=model_cfg.get('gradient_checkpointing', None))
    else:
        warnings.filterwarnings(
            action='ignore',
            message='Torchmetrics v0.9 introduced a new argument class property')
        if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
            raise ValueError(
                f'Not sure how to build model with name={model_cfg.name}')
        return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer)


def build_composer_peft_model(
        pretrained_model_name_or_path: str, lora_args: Dict[str, Any],
        tokenizer: PreTrainedTokenizerBase) -> ComposerHFCausalLM:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            'Error importing from peft. Please verify that peft and peft utils '
            +
            'are installed by running `pip install -e .[peft]` from `llm-foundry/`. '
            + f'Error encountered: {e}')

    # 1) loads a hf model, 2) adds peft modules, 3) wraps it in a ComposerHFCausalLM.
    print('Building Lora config...')
    lora_cfg = LoraConfig(**lora_args)

    print('Building model from HuggingFace checkpoint...')
    model = MPTForCausalLM.from_pretrained(pretrained_model_name_or_path,
                                           trust_remote_code=True)
    print('Model built!')

    print('Adding Lora modules...')
    model = get_peft_model(model, lora_cfg)
    print('Lora modules added!')

    model = ComposerHFCausalLM(model, tokenizer)

    return model


def print_trainable_parameters(model: torch.nn.Module) -> None:
    # Prints the number of trainable parameters in the model.
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )


def build_dataloader(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase,
                     device_batch_size: int):
    if cfg.name == 'text':
        return build_text_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == 'text_denoising':
        return build_text_denoising_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == 'finetuning':
        return build_finetuning_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')


def main(cfg: DictConfig):
    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        'torch.distributed.*_base is a private function and will be deprecated.*'
    )

    # Check for incompatibilities between the model and data loaders
    validate_config(cfg)

    # Resolve all interpolation variables as early as possible
    om.resolve(cfg)

    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(cfg)

    # Get max split size mb
    max_split_size_mb: Optional[int] = cfg.pop('max_split_size_mb', None)
    if max_split_size_mb is not None:
        os.environ[
            'PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_size_mb}'

    # Set seed first
    seed: int = pop_config(cfg, 'seed', must_exist=True)
    reproducibility.seed_all(seed)

    # Initialize pytorch distributed training process groups
    dist_timeout: Union[int, float] = pop_config(cfg,
                                                 'dist_timeout',
                                                 must_exist=False,
                                                 default_value=600.0)
    dist.initialize_dist(get_device(None), timeout=dist_timeout)

    # Get global and device batch size information from distributed/single node setting
    cfg = update_batch_size_info(cfg)
    logged_cfg.update(cfg, merge=True)

    # Mandatory model training configs
    model_config: DictConfig = pop_config(cfg, 'model', must_exist=True)
    tokenizer_config: DictConfig = pop_config(cfg, 'tokenizer', must_exist=True)
    optimizer_config: Dict[str, Any] = pop_config(cfg,
                                                  'optimizer',
                                                  must_exist=True,
                                                  convert=True)
    scheduler_config: Dict[str, Any] = pop_config(cfg,
                                                  'scheduler',
                                                  must_exist=True,
                                                  convert=True)
    train_loader_config: DictConfig = pop_config(cfg,
                                                 'train_loader',
                                                 must_exist=True)

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: Optional[Dict[str, Any]] = pop_config(cfg,
                                                       'fsdp_config',
                                                       must_exist=False,
                                                       default_value=None,
                                                       convert=True)
    lora_config: Optional[Dict[str, Any]] = pop_config(cfg,
                                                       'lora',
                                                       must_exist=False,
                                                       default_value=None,
                                                       convert=True)
    acdc_config: Optional[Dict[str, Any]] = pop_config(cfg,
                                                       'acdc',
                                                       must_exist=False,
                                                       default_value=None,
                                                       convert=True)
    eval_loader_config: Optional[DictConfig] = pop_config(cfg,
                                                          'eval_loader',
                                                          must_exist=False,
                                                          default_value=None)
    icl_tasks_config: Optional[ListConfig] = pop_config(cfg,
                                                        'icl_tasks',
                                                        must_exist=False,
                                                        default_value=None)

    # Optional logging, evaluation and callback configs
    logger_configs: Optional[DictConfig] = pop_config(cfg,
                                                      'loggers',
                                                      must_exist=False,
                                                      default_value=None)
    callback_configs: Optional[DictConfig] = pop_config(cfg,
                                                        'callbacks',
                                                        must_exist=False,
                                                        default_value=None)
    algorithm_configs: Optional[DictConfig] = pop_config(cfg,
                                                         'algorithms',
                                                         must_exist=False,
                                                         default_value=None)

    # Mandatory hyperparameters for training
    device_train_batch_size: int = pop_config(cfg,
                                              'device_train_batch_size',
                                              must_exist=True)
    device_eval_batch_size: int = pop_config(cfg,
                                             'device_eval_batch_size',
                                             must_exist=True)
    max_duration: Union[int, str] = pop_config(cfg,
                                               'max_duration',
                                               must_exist=True)
    eval_interval: Union[int, str] = pop_config(cfg,
                                                'eval_interval',
                                                must_exist=True)
    precision: str = pop_config(cfg, 'precision', must_exist=True)
    max_seq_len: int = pop_config(cfg, 'max_seq_len', must_exist=True)

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get('RUN_NAME', 'llm')
    run_name: str = pop_config(cfg,
                               'run_name',
                               must_exist=False,
                               default_value=default_run_name)
    save_folder: Optional[str] = pop_config(cfg,
                                            'save_folder',
                                            must_exist=False,
                                            default_value=None)
    hf_save_path: Optional[str] = pop_config(cfg,
                                            'hf_save_path',
                                            must_exist=False,
                                            default_value="/mnt/beegfs/alistgrp/mnikdan/llmfoundry-acdc-ckpts/")
    save_latest_filename: str = pop_config(cfg,
                                           'save_latest_filename',
                                           must_exist=False,
                                           default_value='latest-rank{rank}.pt')
    save_overwrite: bool = pop_config(cfg,
                                      'save_overwrite',
                                      must_exist=False,
                                      default_value=False)
    save_weights_only: bool = pop_config(cfg,
                                         'save_weights_only',
                                         must_exist=False,
                                         default_value=False)
    save_filename: str = pop_config(
        cfg,
        'save_filename',
        must_exist=False,
        default_value='ep{epoch}-ba{batch}-rank{rank}.pt')
    save_interval: Union[str, int] = pop_config(cfg,
                                                'save_interval',
                                                must_exist=False,
                                                default_value='1000ba')
    save_num_checkpoints_to_keep: int = pop_config(
        cfg, 'save_num_checkpoints_to_keep', must_exist=False, default_value=-1)
    progress_bar = pop_config(cfg,
                              'progress_bar',
                              must_exist=False,
                              default_value=False)
    log_to_console: bool = pop_config(cfg,
                                      'log_to_console',
                                      must_exist=False,
                                      default_value=True)
    python_log_level: str = pop_config(cfg,
                                       'python_log_level',
                                       must_exist=False,
                                       default_value='debug')
    console_log_interval: Union[int, str] = pop_config(cfg,
                                                       'console_log_interval',
                                                       must_exist=False,
                                                       default_value='1ba')
    device_train_microbatch_size: Union[str, int] = pop_config(
        cfg,
        'device_train_microbatch_size',
        must_exist=False,
        default_value='auto')
    eval_subset_num_batches: int = pop_config(cfg,
                                              'eval_subset_num_batches',
                                              must_exist=False,
                                              default_value=-1)
    train_subset_num_batches: int = pop_config(cfg,
                                              'train_subset_num_batches',
                                              must_exist=False,
                                              default_value=-1)
    eval_first: bool = pop_config(cfg,
                                  'eval_first',
                                  must_exist=False,
                                  default_value=False)
    load_path: str = pop_config(cfg,
                                'load_path',
                                must_exist=False,
                                default_value=None)
    load_weights_only: bool = pop_config(cfg,
                                         'load_weights_only',
                                         must_exist=False,
                                         default_value=False)
    load_ignore_keys: Optional[List[str]] = pop_config(cfg,
                                                       'load_ignore_keys',
                                                       must_exist=False,
                                                       default_value=None)
    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if logged_cfg.get('run_name', None) is not None \
        and save_folder is not None \
        and not save_overwrite \
        and not save_weights_only:
        print('As run_name, save_folder, and save_latest_filename are set, \
                changing autoresume default to True...')
        autoresume_default = True
    autoresume: bool = pop_config(cfg,
                                  'autoresume',
                                  must_exist=False,
                                  default_value=autoresume_default)

    global_train_batch_size = pop_config(cfg, 'global_train_batch_size', must_exist=False)

    # Pop known unused parameters that are used as interpolation variables or
    # created by update_batch_size_info.
    pop_config(cfg, 'data_local', must_exist=False)
    pop_config(cfg, 'data_remote', must_exist=False)
    pop_config(cfg, 'global_seed', must_exist=False)
    pop_config(cfg, 'n_gpus', must_exist=False)
    pop_config(cfg, 'device_train_grad_accum', must_exist=False)

    # Warn users for unused parameters
    for key in cfg:
        warnings.warn(
            f'Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary.'
        )

    # Warn if fsdp is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn(
            'FSDP is not applicable for single-GPU training. Reverting to DDP.')
        fsdp_config = None

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config)
    logged_cfg.update({'fsdp_config': fsdp_config}, merge=True)

    # Build tokenizer
    tokenizer = build_tokenizer(tokenizer_config)

    # Build Model
    print('Initializing model...')
    with init_context:
        if lora_config is not None:  # frozen model + trainable lora modules
            model: ComposerHFCausalLM = build_composer_peft_model(
                model_config.pretrained_model_name_or_path, lora_config['args'],
                tokenizer)
            print_trainable_parameters(model)  # should not be 100%
        else:  # standard model
            model = build_composer_model(model_config, tokenizer)

    # Log number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    logged_cfg.update({'n_params': n_params})

    # Optimizer
    optimizer_name: str = optimizer_config.pop('name')
    optimizer = build_optimizer(model, optimizer_name, optimizer_config)

    # Scheduler
    scheduler_name: str = scheduler_config.pop('name')
    scheduler = build_scheduler(scheduler_name, scheduler_config)

    # Loggers
    loggers = [
        build_logger(str(name), logger_cfg)
        for name, logger_cfg in logger_configs.items()
    ] if logger_configs else None

    # Callbacks
    callbacks = [
        build_callback(str(name), callback_cfg)
        for name, callback_cfg in callback_configs.items()
    ] if callback_configs else None

    # Dataloaders
    print('Building train loader...')
    train_loader = build_dataloader(
        train_loader_config,
        tokenizer,
        device_train_batch_size,
    )

    ## Evaluation
    print('Building eval loader...')
    evaluators = []
    if eval_loader_config is not None:
        assert model.train_metrics is not None
        eval_dataloader = build_dataloader(eval_loader_config, tokenizer,
                                           device_eval_batch_size)
        eval_metric_names = list(model.train_metrics.keys())
        eval_loader = Evaluator(label='eval',
                                dataloader=eval_dataloader,
                                metric_names=eval_metric_names)
        evaluators.append(eval_loader)

    if icl_tasks_config is not None:
        icl_evaluators, _ = build_icl_evaluators(icl_tasks_config, tokenizer,
                                                 max_seq_len,
                                                 device_eval_batch_size)
        evaluators.extend(icl_evaluators)

    # Algorithms
    algorithms = [
        build_algorithm(str(name), algorithm_cfg)
        for name, algorithm_cfg in algorithm_configs.items()
    ] if algorithm_configs else []
    
    # ===== ELDAR: monkey patch ACDC =====
    try:
        # 1-GPU case => len(train_loader.dataset) = 364 868 892  [TRUE SIZE ACCORDING TO HF HUB]
        # 8-GPU case => len(train_loader.dataset) = 45 608 612
        num_ep = float(max_duration.removesuffix('ep'))
        # import os
        # num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        num_steps = int(num_ep * len(train_loader.dataset) / device_train_batch_size)
        print(f"ELDAR DEBUG, num_steps = {num_steps}")
        algorithms.append(
            ACDC(
                model=model,
                **acdc_config,
            )
        )
    except Exception as e:
        print(f'ACDC not added: {e}')
        print(f'[ELDAR debug] len(train_loader.dataset)_on_each_GPU = {len(train_loader.dataset)}, cfg.global_train_batch_size = {global_train_batch_size}')
        raise e
    # ===== ELDAR end =====

    # Build the Trainer
    print('Building trainer...')
    trainer = Trainer(
        run_name=run_name,
        seed=seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=max_duration,
        eval_interval=eval_interval,
        eval_subset_num_batches=eval_subset_num_batches,
        train_subset_num_batches=train_subset_num_batches,
        progress_bar=progress_bar,
        log_to_console=log_to_console,
        console_log_interval=console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=precision,
        algorithms=algorithms,
        device_train_microbatch_size=device_train_microbatch_size,
        fsdp_config=fsdp_config,  # type: ignore
        save_folder=save_folder,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=save_interval,
        save_num_checkpoints_to_keep=save_num_checkpoints_to_keep,
        save_overwrite=save_overwrite,
        save_weights_only=save_weights_only,
        load_path=load_path,
        load_weights_only=load_weights_only,
        load_ignore_keys=load_ignore_keys,
        autoresume=autoresume,
        python_log_level=python_log_level,
        dist_timeout=dist_timeout,
    )

    print('Logging config')
    log_config(logged_cfg)
    torch.cuda.empty_cache()

    # Eval first if requested
    if eval_first and trainer.state.timestamp.batch.value == 0:
        trainer.eval()

    print('Starting training...')
    trainer.fit()

    print('Done.')


    print('Saving directly into HF-friendly format')
    if "WANDB_PROJECT" in os.environ and os.environ["WANDB_DISABLED"] == "False":
        path_to_save = os.path.join(hf_save_path, os.environ["WANDB_PROJECT"], run_name)
    else:
        path_to_save = os.path.join(hf_save_path, run_name)

    if torch.distributed.get_rank() == 0:
        os.makedirs(path_to_save, exist_ok=True) # <-- override if it exists
        
    # Save the model in sharded format
    if fsdp_config is not None:
        with FSDP.summon_full_params(model.model, writeback=False, rank0_only=True, offload_to_cpu=True):
            model.model.save_pretrained(path_to_save, is_main_process=torch.distributed.get_rank() == 0, state_dict=model.model.state_dict())
    else:
        model.model.save_pretrained(path_to_save, is_main_process=torch.distributed.get_rank() == 0, state_dict=model.model.state_dict())
    
    if torch.distributed.get_rank() == 0:
        tokenizer.save_pretrained(path_to_save)
    # NOTE: for some reason the saving code above would create empty pytorch_model.bin file, so we delete it manually
    # TODO: figure out why this happens
    if torch.distributed.get_rank() == 0 and os.path.exists(os.path.join(path_to_save, "pytorch_model.bin")):
        tmp = torch.load(os.path.join(path_to_save, "pytorch_model.bin"))
        if not tmp:  # empty dict, remove it
            os.remove(os.path.join(path_to_save, "pytorch_model.bin"))

    print('Done.')

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    assert isinstance(cfg, DictConfig)
    main(cfg)
