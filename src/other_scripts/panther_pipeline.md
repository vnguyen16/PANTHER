# Step 1. Prototype construction
*2.5x*
python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_pt --split_dir cross-val\FA_PT_k=4 --split_names train --in_dim 1024 --n_proto_patches 1000000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10

*5x* 
python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/uniextracted_mag5x_patch224_fp/feats_pt --split_dir cross-val\FA_PT_k=4 --split_names train --in_dim 1024 --n_proto_patches 1000000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10

*10x*
python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_10x/uniextracted_mag10x_patch224_fp/feats_pt --split_dir cross-val\FA_PT_k=4 --split_names train --in_dim 1024 --n_proto_patches 500000 --n_proto 16 --n_init 3 --seed 1 --num_workers 10

- changed n_init to 3 ^

*SPIDER dataset*
python -m training.main_prototype --mode kmeans --data_source C:\Users\Vivian\Documents\CLAM\CLAM\FEATURES_DIR_5x\spider_run2\feats_pt --split_dir spider\FA_PT_k=4 --split_names train --in_dim 1024 --n_proto_patches 1000000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10

- change split dir, data source
- change normalization of features in clustering script

# Step 2B. Training downstream model

*ABMIL*
python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type allcat --data_source C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_pt --split_dir cross-val\FA_PT_k=0 --proto_path splits/cross-val/FA_PT_k=0/prototypes --in_dim 1024 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code cross-val/2.5x_ABMIL --emb_model_type LinearEmb

*OT*
python -m training.main_classification --task fa_vs_pt --model_type OT --model_config OT_default --out_type allcat --data_source C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_pt --split_dir cross-val\FA_PT_k=0 --proto_path splits/cross-val/FA_PT_k=0/prototypes --in_dim 1024 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code cross-val/2.5x_OT --emb_model_type LinearEmb


*OT* --> trying with unnoramlized protos --> also doesn't work
python -m training.main_classification --task fa_vs_pt --model_type OT --model_config OT_default --out_type allcat --data_source C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_pt --split_dir FA_PT_2.5x_k=0 --proto_path splits/FA_PT_2.5x_k=0/prototypes --in_dim 1024 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code 2.5x_OT --emb_model_type LinearEmb

*2.5x*
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_pt --split_dir cross-val\FA_PT_k=4 --proto_path splits/cross-val/FA_PT_k=4/prototypes --in_dim 1024 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code cross-val/2.5x_norm_avgmean --emb_model_type LinearEmb

*5x*
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/uniextracted_mag5x_patch224_fp/feats_pt --split_dir cross-val\FA_PT_k=0 --proto_path splits/cross-val/FA_PT_k=0/prototypes --in_dim 1024 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code cross-val/5x_norm_avgmean --emb_model_type LinearEmb

*10x*
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_10x/uniextracted_mag10x_patch224_fp/feats_pt --split_dir cross-val\FA_PT_k=4 --proto_path splits/cross-val/FA_PT_k=4/prototypes --in_dim 1024 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code cross-val/10x_norm_avgmean --emb_model_type LinearEmb

- change split dir, data source, proto path, out type, exp code, model type (optional), model config
- out type [allcat, weight avg mean, ]
- model type [PANTHER, ABMIL, H2T]

- created create_kfold_splits.ipynb to make new cross val splits and also average out metrics for each fold

# Step 3. Visualization
use src\visualization\prototypical_assignment_map_visualization.ipynb

