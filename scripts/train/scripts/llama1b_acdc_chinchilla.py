import fire
import inspect
import random
import os

def main(
        s=0.90, # sparsity
        m='vanilla', # method: vanilla or gradual
        g=True, # global
        st='unstr', # structure
        bs=64, # batch_size
        e=1, # num_epochs
        aw=400, # acdc_warmup
        apl=50, # acdc_phase_len
        alsl=100, # acdc_last_sparse_len
        aldl=100, # acdc_last_dense_len
        sa=False, # scale_acdc_as_well
        p='magnitude', # pruner
        wd=0, # weight_decay
        o='decoupled_adamw', # optim
        lr=0.0001,
        d=None, # devices
        dpath=None, # data_path
    ):
    if d is None:
        d = [str(i) for i in range(8)]
    
    args = [v.split('=')[0].strip('(').strip(')').strip() for v in str(inspect.signature(main)).split(',')]
    ls = locals()
    args_dict = {arg:ls[arg] for arg in args if arg not in ['d', 'dpath']}
    run_name = f'chinch-llama_1b-c4-acdc-{"-".join([f"{key}_{value}" for key, value in args_dict.items()])}-{random.randint(10000, 99999)}'
    
    acdc_scale = e if sa else 1
    # 8 for 1B/125M (model size) and 4 for 4096/1024 (seq len)
    num_total_steps = int(4 * 8 * 1200 * (512 / bs) * e)
    params = {
        'eval_subset_num_batches': int(4 * 8 * 100 * (512 / bs)),
        'train_subset_num_batches': int(4 * 8 * 1200 * (512 / bs)),
        'max_duration': f'{e}ep',
        'eval_interval': f'{int(4 * 8 * 200 * (512 / bs) * e)}ba', # 6 evals during training
        'global_train_batch_size': bs, # should divide 4 * num_devices
        'acdc.is_global': g,
        'acdc.pruner': p,
        'acdc.sparsity_structure': st,
        'acdc_final_sparsity': s,
        'acdc_schedule_0_end': aw * acdc_scale, # first dense (warmup)
        'acdc_schedule_1_freq': apl * acdc_scale, # the dense/high phase frequency
        'acdc_schedule_1_end': num_total_steps - (alsl + aldl) * acdc_scale, # the dense/high phase end
        'acdc_schedule_2_end': num_total_steps - alsl * acdc_scale, # the longer last dense (after this the model is always sparse)
        'optimizer.weight_decay': wd,
        'optimizer.name': o,
        'optimizer.lr': lr,
        'data_local': dpath,
        'run_name': run_name,
        'hf_save_path': './checkpoints/',
    }

    if dpath.startswith('gs://') or dpath.startswith('http'):
        del params['data_local']
        params['data_remote'] = dpath

    print(run_name)
    print(params)

    command = f'CUDA_VISIBLE_DEVICES={",".join(d)} composer train_acdcpp.py yamls/acdc_levanter/llama1b_acdc_{m}.yaml'
    for key, value in params.items():
        command += f' {key}={value}'
    print(command)
    os.system(command)

if __name__ == '__main__':
    fire.Fire(main)
