import copy
import os

from griddle.griddle_experiment_config import (
    ExperimentConfigGrid,
    GriddleMode,
)
from griddle.utils import param_grid
from griddle.utils import is_aws_cluster
# -----
from train_experiment import ExperimentConfig
from omegaconf import OmegaConf


# -----

DEFAULT_STATS = ["Racc_5", "Racc_15", "Racc_30","Tacc_5", "Tacc_15", "Tacc_30"]



if is_aws_cluster:
    EXPERIMENT_ROOT = os.path.expandvars("/fsx-repligen/$USER/gridexp/")
else:
    EXPERIMENT_ROOT = os.path.expandvars("/checkpoint/$USER/exps/griddle/")


def griddle_zoo_configs(
    griddle_mode: GriddleMode,
    experiment_mode: str,
):
    # ----
    cfgs = {}
    # ----


    JOB_PARAMS = {
        "slurm_time": 4000,
        "slurm_gpus_per_node": 8,
        "slurm_partition": "learngenai",
        # "slurm_cpus_per_task": 80,  
        "slurm_cpus_per_gpu": 12,  
        "slurm_mem": "0",
        "slurm_exclusive": True,
    }

    accelerate_args = {
        "num_machines": 1,
        "multi_gpu": True,
        "debug": True,              # if met error, will give sth; if set to False, give nothing with error
        "mixed_precision": "no",
        "num_cpu_threads_per_process": 8,
        "num_processes": 8,         # 4 gpus requested
    }
    
    # griddle_run griddle_zoo_configs_folder=./griddle_zoo_configs cfg="preload" griddle_mode=DISPATCH                                                                                                                                                   

    is_eval = experiment_mode == "eval"
    is_debug = griddle_mode == GriddleMode.DEBUG

    # cfg_name = "train001"
    
    cfg_name = "preload"
    hydra_config = "../cfgs/default_train.yaml"
    base_conf = OmegaConf.load(hydra_config)
    
    # Common params
    base_conf.experiment_mode = experiment_mode
    # base_conf.ckpt = "tmp/co3d_model_Apr16.pth"
    # base_conf.GGS.enable = False
    base_conf.cfg_name = cfg_name
    
    if is_debug:
        base_conf = base_conf
        base_conf.debug = True


        JOB_PARAMS = {
            "slurm_time": 4000,
            "slurm_gpus_per_node": 1,
            "slurm_partition": "learngenai",
            "slurm_cpus_per_task": 10,     
            "slurm_mem": "0",
        }
        
        accelerate_args = {
            "num_machines": 1,
            "multi_gpu": True,
            "debug": True,              # if met error, will give sth; if set to False, give nothing with error
            "mixed_precision": "no",
            "num_processes": 1,         # 4 gpus requested
        }
    


    if is_eval:
        # base_conf.update({})
        base_conf = base_conf
        
    grid_param = {
        "train.lr": [0.001],
        # "train.category": ["debug"],
        # "lr": [0.0001, 0.0005],
        "train.num_workers": [0, 4],
        "train.preload_image": [True],
        # "train.pin_memory"
        # "train.max_images": [180],
        # "train.pt3d_co3d": [True, False]
        # "train.dynamic_batch": [True, False]
        # "train.normalize_cameras": [True, False]
    }
    
    
    grid, exp_names = param_grid(grid_param, common_params=base_conf, return_name = True)
    exp_names = [cfg_name + "/" + name for name in exp_names]

    cfgs[cfg_name] = ExperimentConfigGrid(
        griddle_experiment_configurable=ExperimentConfig(),
        cfg_dicts=grid,
        exp_names=exp_names,
        experiment_root=EXPERIMENT_ROOT,
        experiment_name_prefix=cfg_name,
        experiment_mode=experiment_mode,
        autogenerate_exp_dirs=True,
        stats_analyze=copy.deepcopy(
            DEFAULT_STATS_EVAL if is_eval else DEFAULT_STATS
        ),
        slurm_job_params=copy.deepcopy(JOB_PARAMS),
        accelerate_job_params = copy.deepcopy(accelerate_args), 
    )
    #################################


    return cfgs


