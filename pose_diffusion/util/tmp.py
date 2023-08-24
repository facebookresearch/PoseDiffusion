# Reproducing CO3Dv2 Implicitron experiments. It lives here to avoid syncing from fbcode.
# Run using Griddle from pixar_replay root, e.g.:
# > griddle_run griddle_zoo_configs_folder=pixar_replay/experimental/projects/implicitron/griddle_zoo_configs cfg=singleseq_regression_griddle_pr griddle_mode=DEBUG

import copy
import os
from collections import defaultdict
from typing import Any, Dict, Sequence

from griddle.griddle_experiment_config import (
    ExperimentConfigGrid,
    GriddleMode,
)



from griddle.utils import param_grid, get_visdom_server
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider import CO3D_CATEGORIES
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    _CO3DV2_DATASET_ROOT,
    JsonIndexDatasetMapProviderV2,  # noqa
)

from pixar_replay.experimental.models.jaypose.experiment_config import (
    ImplicitronExperimentConfig,
)
from jay_utils import *
EXPERIMENT_ROOT = os.path.expandvars("/checkpoint/$USER/exps/jaypose")
# TODO: this should be configurable!

DATASET_ARGS = "data_source_ImplicitronDataSource_args.dataset_map_provider_JsonIndexDatasetMapProviderV2_args"
DATA_LOADER_ARGS = "data_source_ImplicitronDataSource_args.data_loader_map_provider_SequenceDataLoaderMapProvider_args"
TRAINING_ARGS = "training_loop_PoseDiffusionTrainingLoop_args"
MODEL_ARGS = "model_factory_ImplicitronModelFactory_args.model_PoseDiffusionModel_args"
OPTIMIZER_ARGS = "optimizer_factory_ImplicitronOptimizerFactory_args"

expand_args_fields(JsonIndexDatasetMapProviderV2)

def griddle_zoo_configs(
    griddle_mode: GriddleMode,
    experiment_mode: str,
) -> Dict[str, ExperimentConfigGrid]:
    # ----
    cfgs = {}
    # ----

    griddle_experiment_configurable = ImplicitronExperimentConfig(
        experiment_mode=experiment_mode,
    )

    is_test = experiment_mode == "test"
    is_eval = experiment_mode == "eval"
    is_debug = griddle_mode == GriddleMode.DEBUG

    shared_params: Dict[str, Any] = {
        "data_source_ImplicitronDataSource_args.dataset_map_provider_class_type": "JsonIndexDatasetMapProviderV2",
        "model_factory_ImplicitronModelFactory_args.model_class_type": "PoseDiffusionModel",
        "training_loop_class_type": "PoseDiffusionTrainingLoop",
        f"{TRAINING_ARGS}.store_checkpoints": True,
        f"{TRAINING_ARGS}.visdom_server": get_visdom_server(),
        f"{TRAINING_ARGS}.metric_print_interval": 50,
        f"{TRAINING_ARGS}.visualize_interval": -1,
        f"{TRAINING_ARGS}.max_epochs": 300,
        f"{TRAINING_ARGS}.clip_grad": 1.0,                    
        f"{TRAINING_ARGS}.store_checkpoints_purge": -1,                        
        f"{TRAINING_ARGS}.evaluator_ImplicitronEvaluator_args.is_multisequence": True,
        ###
        f"{MODEL_ARGS}.v3loss_in_optform": True,           
        f"{MODEL_ARGS}.z_predictor_GlobalImFeatureZPredictor_args.l2_normalize": False,
        ###
        f"{DATA_LOADER_ARGS}.batch_size": 180,     
        f"{DATA_LOADER_ARGS}.dataset_length_train": 18432,
        f"{DATA_LOADER_ARGS}.dataset_length_val": 768,
        f"{DATA_LOADER_ARGS}.num_workers": 4,
        f"{DATA_LOADER_ARGS}.test_conditioning_type": "SAME",
        ###
        f"{DATASET_ARGS}.dataset_root": _CO3DV2_DATASET_ROOT,
        f"{DATASET_ARGS}.dataset_JsonIndexDatasetV2_args.image_height": 224,
        f"{DATASET_ARGS}.dataset_JsonIndexDatasetV2_args.image_width": 224,
        f"{DATASET_ARGS}.test_on_train": False,
        f"{DATASET_ARGS}.dataset_JsonIndexDatasetV2_args.limit_sequences_to": -1,
        f"{DATASET_ARGS}.category": "donut,apple,hydrant,vase,cake,bench,teddybear,plant,broccoli,orange,toytruck,toytrain,mouse,toaster,bowl,bottle,carrot,motorcycle,car,keyboard,chair,handbag,toybus,toyplane,backpack,parkingmeter,cup,baseballglove,stopsign,laptop,wineglass,umbrella,pizza,hairdryer,toilet,cellphone,tv,microwave,bicycle,banana,baseballbat",
    }

    if is_debug:
        shared_params = {
            **shared_params,
            "seed": 0,
            f"{MODEL_ARGS}.is_debug": is_debug,
            f"{DATA_LOADER_ARGS}.batch_size": 80,     
            f"{DATA_LOADER_ARGS}.dataset_length_train": 2, 
            f"{DATA_LOADER_ARGS}.dataset_length_val": 2,
            f"{TRAINING_ARGS}.metric_print_interval": 1,
            f"{TRAINING_ARGS}.store_checkpoints": False,
            f"{TRAINING_ARGS}.visdom_env": "debug",
            f"{TRAINING_ARGS}.visdom_port": os.getenv("VISDOM_PORT", 8097),
            f"{DATA_LOADER_ARGS}.num_workers": 2,
            f"{DATASET_ARGS}.category": "donut",
        }

    ##########################################

    JOB_PARAMS = {
        "slurm_time": 4000,
        "slurm_gpus_per_node": 8,
        # "slurm_partition": "app1",
        "slurm_partition": "learnaccel",
        "slurm_cpus_per_task": 80,     
        "slurm_constraint": "volta32gb",   
        "slurm_mem": "0",
    }
    
    accelerate_args = {
        "num_machines": 1,
        # "multi_gpu": True,
        "debug": True,              # if met error, will give sth; if set to False, give nothing with error
        "mixed_precision": "no",
        "num_processes": 8,         # 4 gpus requested
    }
    
    ##################################################################################################

    grid_param = {
        # Varying
        ########
        f"{MODEL_ARGS}.relative_to_first_camera": [True],  
        f"{MODEL_ARGS}.color_aug": [True],        # batch color jitter
        f"{MODEL_ARGS}.erase_aug": [False],       # batch erase
        f"{MODEL_ARGS}.T": [100],
        f"{MODEL_ARGS}.per_scene_acc": [True],
        f"{MODEL_ARGS}.z_predictor_GlobalImFeatureZPredictor_args.freeze": [False],
        f"{DATASET_ARGS}.dataset_JsonIndexDatasetV2_args.crop_wo_mask": [True],
        }
    
    grid_param = {
        **grid_param, 
        ################################
        f"{MODEL_ARGS}.batch_repeat": [64],    
        f"{MODEL_ARGS}.use_tvec_s": [True],    
        f"{MODEL_ARGS}.optform_type": ["quaR_absT_fl"],
        f"{DATASET_ARGS}.subset_name": ["fewview_dev"],
        #####
        f"{TRAINING_ARGS}.traintime_eval": [True],   
        f"{TRAINING_ARGS}.validation_interval": [5],                      
        f"{MODEL_ARGS}.loss_weights.loss_center": [0.0],
        f"{MODEL_ARGS}.loss_weights.loss_diffusion": [1.0],
        f"{MODEL_ARGS}.tvec_s_type": ["seq_median_norm"],    
        f"{MODEL_ARGS}.random_tvec_aug": [False],    
        f"{MODEL_ARGS}.cond_score_model_args.norm_first": [True],     
        f"{MODEL_ARGS}.normalize_predT": [False],    
        f"{MODEL_ARGS}.normalize_predT_sampling": [False],    
        f"{MODEL_ARGS}.normalize_predT_detach": [False],    
        f"{MODEL_ARGS}.beta_schedule":["custom"],
        f"{DATASET_ARGS}.dataset_JsonIndexDatasetV2_args.box_random_aug": [False], # True
        f"{MODEL_ARGS}.z_predictor_GlobalImFeatureZPredictor_args.multiscale": [True],
        f"{MODEL_ARGS}.z_predictor_GlobalImFeatureZPredictor_args.name":  ["dino_vits16"],
        f"{MODEL_ARGS}.cond_score_model_args.num_encoder_layers": [2],  
        f"{MODEL_ARGS}.cond_score_model_args.num_decoder_layers": [6], 
        ##############################################################################
        # f"{DATA_LOADER_ARGS}.batch_size": [180],
        f"{MODEL_ARGS}.cond_score_model_args.harmonic": [True], 
        f"{MODEL_ARGS}.cond_score_model_args.last_type": [6],            #  
        # f"{DATASET_ARGS}.dataset_JsonIndexDatasetV2_args.box_crop": [True, False],
        f"{DATASET_ARGS}.dataset_JsonIndexDatasetV2_args.box_crop": [True],
        f"{OPTIMIZER_ARGS}.lr": [0.000025, 0.00005, 0.000075], 
        f"{MODEL_ARGS}.cond_score_model_args.nhead": [4], 
        f"{MODEL_ARGS}.cond_score_model_args.time_embed_flag": [True], 
        f"{MODEL_ARGS}.beta_T":[0.1],    
        f"{OPTIMIZER_ARGS}.weight_decay": [0.0],
        f"{DATA_LOADER_ARGS}.images_per_seq_options": ["list(range(2, 51))", "list(range(3, 51))"],
        f"{OPTIMIZER_ARGS}.lr_policy": ["WarmupCosineRestart", "CosineRestart"],
        f"{OPTIMIZER_ARGS}.restart_num": [50],
        ################################
    }
        
    grid_param = {
        **grid_param, 
        ################################
        f"{MODEL_ARGS}.pred_objective": ["pred_noise"],
        f"{MODEL_ARGS}.cond_score_model_args.residual_predict": [False],}
    
    

    ##########################
    JOB_PARAMS = {
        "slurm_time": 4000,
        "slurm_partition": "learnaccel",
        "slurm_gpus_per_node": 1,
        "slurm_cpus_per_task": 5,        
        "slurm_mem": "32G",
    }
    
    accelerate_args = {
        "num_machines": 1,
        # "multi_gpu": False,
        "debug": True,              # if met error, will give sth; if set to False, give nothing with error
        "mixed_precision": "no",
        "num_processes": 1,         # 4 gpus requested
    }
    
    grid_param = {
        **grid_param,
        #####
        f"{TRAINING_ARGS}.validate_only": [True],
        f"{TRAINING_ARGS}.colmap_test": [False],
        f"{TRAINING_ARGS}.random_order": [True],
        f"{TRAINING_ARGS}.test_num_frame": [20, 10, 5, 3],
        f"{TRAINING_ARGS}.test_as_rel_pose": [True],
        f"{DATASET_ARGS}.subset_name": ["fewview_dev"],
        f"{TRAINING_ARGS}.align_to_first": [False],
        f"{TRAINING_ARGS}.align_to_gt_pose": ["7dof"],
        }
    shared_params[f"{DATASET_ARGS}.category"] = "donut"
    

    grid_param[f"{OPTIMIZER_ARGS}.lr_policy"] = ["WarmupCosineRestart"]
    grid_param[f"{OPTIMIZER_ARGS}.lr"] = [0.000075]
    grid_param[f"{DATA_LOADER_ARGS}.images_per_seq_options"] = ["list(range(3, 51))"]
    
    base_str = "/checkpoint/jianyuan/exps/jaypose/A010/A010_opt_lr_7_5e_05_d_d_images_per_seq_options_list(range({:01d}__51))_opt_lr_policy_WarmupCosineRestart/model_epoch_{:08d}.pth"
    result = [base_str.format(restart_num, i) for restart_num in (3, 2) for i in range(80, 200)]
    grid_param["model_factory_ImplicitronModelFactory_args.resume_path"] = result

    
    
    cfg_dicts = []
    cfg_dicts.extend(param_grid(grid_param, common_params=shared_params))
    cfg_name = "A010camera"
    cfgs[cfg_name] = ExperimentConfigGrid(
        griddle_experiment_configurable=griddle_experiment_configurable,
        cfg_dicts=cfg_dicts,
        experiment_root=os.path.join(EXPERIMENT_ROOT, cfg_name),
        experiment_name_prefix=cfg_name,
        experiment_mode=experiment_mode,
        autogenerate_exp_dirs=True,
        stats_analyze=DEFAULT_STATS,
        slurm_job_params=copy.deepcopy(JOB_PARAMS),
        accelerate_job_params = copy.deepcopy(accelerate_args), )
    

    grid_param = {
        **grid_param, 
        ################################
        f"{MODEL_ARGS}.cond_score_model_args.num_encoder_layers": [8],  
        f"{MODEL_ARGS}.cond_score_model_args.num_decoder_layers": [0], 
        }
    
    # base_str = "/checkpoint/jianyuan/exps/jaypose/A010_onlyenc/A010_onlyenc_opt_lr_7_5e_05_d_d_images_per_seq_options_list(range({:01d}__51))_opt_lr_policy_WarmupCosineRestart/model_epoch_{:08d}.pth"
    base_str = "/checkpoint/jianyuan/CR/model_epoch_{:08d}.pth"

    result = [base_str.format(i)  for i in range(90, 200)]
    grid_param["model_factory_ImplicitronModelFactory_args.resume_path"] = result

    grid_param["model_factory_ImplicitronModelFactory_args.resume_path"] = ["/checkpoint/jianyuan/CR/model_epoch_00000150.pth"]


    cfg_dicts = []
    cfg_dicts.extend(param_grid(grid_param, common_params=shared_params))
    cfg_name = "camR"
    cfgs[cfg_name] = ExperimentConfigGrid(
        griddle_experiment_configurable=griddle_experiment_configurable,
        cfg_dicts=cfg_dicts,
        experiment_root=os.path.join(EXPERIMENT_ROOT, cfg_name),
        experiment_name_prefix=cfg_name,
        experiment_mode=experiment_mode,
        autogenerate_exp_dirs=True,
        stats_analyze=DEFAULT_STATS,
        slurm_job_params=copy.deepcopy(JOB_PARAMS),
        accelerate_job_params = copy.deepcopy(accelerate_args), )


    grid_param["model_factory_ImplicitronModelFactory_args.resume_path"] = ["/checkpoint/jianyuan/CR/model_epoch_00000150.pth"]

    grid_param[f"{TRAINING_ARGS}.out_exp_dir"] = ["/checkpoint/jianyuan/exps/jaypose/SavecamRGGS"]

    grid_param[f"{MODEL_ARGS}.fcons_sampling"] = [True] 
    grid_param[f"{MODEL_ARGS}.GGS_lr"] = [1e-2] 
    grid_param[f"{MODEL_ARGS}.GGS_clip_grad"] = [-1.0] 
    grid_param[f"{TRAINING_ARGS}.load_col_matches"] = [True]   


    grid_param[f"{MODEL_ARGS}.GGS_RT_iter_refine"] = [3]
    grid_param[f"{MODEL_ARGS}.GGS_change_SR"] = [-1]
    grid_param[f"{MODEL_ARGS}.GGS_change_ST"] = [-1]

    grid_param[f"{MODEL_ARGS}.GGS_final"] = [True] 
    grid_param[f"{MODEL_ARGS}.GGS_torch_opt"] = [True] 
    grid_param[f"{MODEL_ARGS}.GGS_clip_grad_by_pred"] = [True]
    grid_param[f"{MODEL_ARGS}.GGS_clip_grad_by_pred_type"] = [1]

    grid_param[f"{MODEL_ARGS}.GGS_use_fl"] = [2] 
    grid_param[f"{MODEL_ARGS}.GGS_loss_type"] = ["l1"] 
    grid_param[f"{MODEL_ARGS}.GGS_sampson_max"] = [10] 
    grid_param[f"{MODEL_ARGS}.GGS_min_matches"] = [10] 

    grid_param[f"{MODEL_ARGS}.GGS_late"] = [True] 
    grid_param[f"{MODEL_ARGS}.GGS_iter_final"] = [300]
    grid_param[f"{MODEL_ARGS}.GGS_iter"] = [350]
    grid_param[f"{MODEL_ARGS}.GGS_start_step"] = [10]

    grid_param[f"{MODEL_ARGS}.max_GGS_change_scale"]  = [0.0001]
    grid_param[f"{TRAINING_ARGS}.load_outputs"] = [True]    
    grid_param[f"{TRAINING_ARGS}.save_outputs"] = [True]    

    grid_param[f"{TRAINING_ARGS}.col_matches_dir"] = ['co3dv2_few_spsg']

    grid_param[f"{TRAINING_ARGS}.test_num_frame"] = [20, 10, 5, 3]


    # grid_param[f"{TRAINING_ARGS}.split_eval_seq_names"]= [60]
    # grid_param[f"{TRAINING_ARGS}.split_eval_seq_names_n"] = list(range(0,60))
    
    grid_param[f"{TRAINING_ARGS}.split_eval_seq_names"]= [20]
    grid_param[f"{TRAINING_ARGS}.split_eval_seq_names_n"] = list(range(0,20))
    

    cfg_dicts = []
    cfg_dicts.extend(param_grid(grid_param, common_params=shared_params))
    cfg_name = "camRGGSsec"
    cfgs[cfg_name] = ExperimentConfigGrid(
        griddle_experiment_configurable=griddle_experiment_configurable,
        cfg_dicts=cfg_dicts,
        experiment_root=os.path.join(EXPERIMENT_ROOT, cfg_name),
        experiment_name_prefix=cfg_name,
        experiment_mode=experiment_mode,
        autogenerate_exp_dirs=True,
        stats_analyze=DEFAULT_STATS,
        slurm_job_params=copy.deepcopy(JOB_PARAMS),
        accelerate_job_params = copy.deepcopy(accelerate_args), )


    grid_param[f"{TRAINING_ARGS}.test_num_frame"] = [10, 5, 3]
    grid_param[f"{TRAINING_ARGS}.split_eval_seq_names"]= [30]
    grid_param[f"{TRAINING_ARGS}.split_eval_seq_names_n"] = list(range(0,30))

    cfg_dicts = []
    cfg_dicts.extend(param_grid(grid_param, common_params=shared_params))
    cfg_name = "camRGGSfillAga"
    cfgs[cfg_name] = ExperimentConfigGrid(
        griddle_experiment_configurable=griddle_experiment_configurable,
        cfg_dicts=cfg_dicts,
        experiment_root=os.path.join(EXPERIMENT_ROOT, cfg_name),
        experiment_name_prefix=cfg_name,
        experiment_mode=experiment_mode,
        autogenerate_exp_dirs=True,
        stats_analyze=DEFAULT_STATS,
        slurm_job_params=copy.deepcopy(JOB_PARAMS),
        accelerate_job_params = copy.deepcopy(accelerate_args), )


    grid_param[f"{TRAINING_ARGS}.test_num_frame"] = [20, 10, 5, 3]
    grid_param[f"{TRAINING_ARGS}.split_eval_seq_names"]= [-1]
    grid_param[f"{TRAINING_ARGS}.split_eval_seq_names_n"] = [0]


    cfg_dicts = []
    cfg_dicts.extend(param_grid(grid_param, common_params=shared_params))
    cfg_name = "camRGGSCheck"
    cfgs[cfg_name] = ExperimentConfigGrid(
        griddle_experiment_configurable=griddle_experiment_configurable,
        cfg_dicts=cfg_dicts,
        experiment_root=os.path.join(EXPERIMENT_ROOT, cfg_name),
        experiment_name_prefix=cfg_name,
        experiment_mode=experiment_mode,
        autogenerate_exp_dirs=True,
        stats_analyze=DEFAULT_STATS,
        slurm_job_params=copy.deepcopy(JOB_PARAMS),
        accelerate_job_params = copy.deepcopy(accelerate_args), )



    return cfgs
