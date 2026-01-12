build_h5_concat_feats.py

other_scripts\multiscale\build_h5_concat_feats.py

python -c "import glob, os; from other_scripts/multiscale/build_h5_concat_feats.py import build_panther_h5; in_dir='C:/Users/Vivian/Documents/dsmil-wsi/temp_train'; out_dir='C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/unifiltextracted_mag2x_patch224_fp'; os.makedirs(out_dir, exist_ok=True); [build_panther_h5(f, os.path.join(coord_dir, os.path.splitext(os.path.basename(f))[0]+'_2.5x.h5'), os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0]+'_multiscale.h5'), coords_from='h5') for f in glob.glob(os.path.join(in_dir, '*.pt'))]"

python -c "import sys, os, glob; sys.path.append('C:/Users/Vivian/Documents/PANTHER/PANTHER/src'); from other_scripts.multiscale.build_h5_concat_feats import build_panther_h5; in_dir='C:/Users/Vivian/Documents/dsmil-wsi/temp_train'; coord_dir='C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_h5'; out_dir='C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/unifiltextracted_mag2x_patch224_fp'; os.makedirs(out_dir, exist_ok=True); [build_panther_h5(f, os.path.join(coord_dir, os.path.splitext(os.path.basename(f))[0]+'.h5'), os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0]+'.h5'), coords_from='h5') for f in glob.glob(os.path.join(in_dir, '*.pt'))]"

# 1) generate h5 files with coords using concat pt feat files from dsmil
run cells in:
> read_feats.ipynb

# 2) Prototype construction
# covert dict .pt to raw tensor .pt 

python -c "import os, glob, torch; src=r'C:\Users\Vivian\Documents\PANTHER\PANTHER\features\multiscale\52\uniextracted_mag52_patch224_fp\feats_pt1';dst=r'C:\Users\Vivian\Documents\PANTHER\PANTHER\features\multiscale\52\uniextracted_mag52_patch224_fp\feats_pt'; os.makedirs(dst, exist_ok=True); [torch.save((lambda o: o['features'] if isinstance(o,dict) else o)(torch.load(p, map_location='cpu')), os.path.join(dst, os.path.basename(p))) for p in glob.glob(os.path.join(src, '*.pt'))]"

2.5 + 5 = C:\Users\Vivian\Documents\PANTHER\PANTHER\features\multiscale\uniextracted_mag25x_patch224_fp
2.5 + 10 = C:\Users\Vivian\Documents\PANTHER\PANTHER\features\multiscale\uniextracted_mag210x_patch224_fp

python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag25x_patch224_fp/feats_pt --split_dir multiscale/25_norm_patient/FA_PT_k=4 --split_names train --in_dim 2048 --n_proto_patches 400000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10 --sample_col case_id

python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/210/uniextracted_mag210x_patch224_fp/feats_pt --split_dir multiscale/210_run2_patient/FA_PT_k=4 --split_names train --in_dim 2048 --n_proto_patches 400000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10 --sample_col case_id

> 5_2
python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_patient/FA_PT_k=2 --split_names train --in_dim 2048 --n_proto_patches 400000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10 --sample_col case_id

> changed n patches to 400000 from 1000000
> change data_source amd split dir


# 3)  Training downstream model
- CHANGED in_dim in LinearEmb config and PANTHER_fa_pt

# PANTHER

python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type allcat --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag210x_patch224_fp/feats_pt --split_dir multiscale\210\FA_PT_k=0 --proto_path splits/multiscale/210/FA_PT_k=0/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_10_avgmean --emb_model_type LinearEmb

> 2_10
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type allcat --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag210x_patch224_fp/feats_pt --split_dir multiscale\210_patient\FA_PT_k=4 --proto_path splits/multiscale/210_patient/FA_PT_k=4/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_10_patient_allcat --emb_model_type LinearEmb --sample_col case_id

python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type allcat --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag210x_patch224_fp/feats_pt --split_dir multiscale\210\FA_PT_k=0 --proto_path splits/multiscale/210/FA_PT_k=0/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_10_ABMIL --emb_model_type LinearEmb --sample_col slide_id

> 210 run 2
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type allcat --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/210/uniextracted_mag210x_patch224_fp/feats_pt --split_dir multiscale\210_run2_patient\FA_PT_k=4 --proto_path splits/multiscale/210_run2_patient/FA_PT_k=4/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/210_run2_patient_allcat --emb_model_type LinearEmb --sample_col case_id

> 2_5
- allcat
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type allcat --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag25x_patch224_fp/feats_pt --split_dir multiscale\25_norm_patient\FA_PT_k=0 --proto_path splits/multiscale/25_norm_patient/FA_PT_k=0/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_5_norm_patient_allcat --emb_model_type LinearEmb --sample_col case_id

- avgmean
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag25x_patch224_fp/feats_pt --split_dir multiscale\25_norm_patient\FA_PT_k=4 --proto_path splits/multiscale/25_norm_patient/FA_PT_k=4/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_5_norm_patient_avgmean --emb_model_type LinearEmb --sample_col case_id

- ABMIL
python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag25x_patch224_fp/feats_pt --split_dir multiscale\25_patient\FA_PT_k=4 --proto_path splits/multiscale/25_patient/FA_PT_k=4/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_5_patient_ABMIL --emb_model_type LinearEmb --sample_col case_id

> 5_2
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type allcat --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_patient/FA_PT_k=1 --proto_path splits/multiscale/52_patient/FA_PT_k=1/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_patient_allcat --emb_model_type LinearEmb --sample_col case_id

python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type allcat --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52/FA_PT_k=4 --proto_path splits/multiscale/52/FA_PT_k=4/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/5_2_ABMIL --emb_model_type LinearEmb --sample_col slide_id

# weight avg mean
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag2x_patch224_fp/feats_pt --split_dir multiscale\FA_PT_k=0 --proto_path splits/multiscale/FA_PT_k=0/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/avg_mean --emb_model_type LinearEmb

- normalized multi features
python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag210x_patch224_fp/feats_pt --split_dir multiscale\210_norm\FA_PT_k=4 --proto_path splits/multiscale/210_norm/FA_PT_k=4/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_10_norm_avgmean --emb_model_type LinearEmb

python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag210x_patch224_fp/feats_pt --split_dir multiscale\210_patient\FA_PT_k=0 --proto_path splits/multiscale/210_patient/FA_PT_k=0/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_10_patient_avgmean --emb_model_type LinearEmb --sample_col case_id

# ABMIL
python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type allcat --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag2x_patch224_fp/feats_pt --split_dir multiscale\FA_PT_k=4 --proto_path splits/multiscale/FA_PT_k=4/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_5_ABMIL --emb_model_type LinearEmb

python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type allcat --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/uniextracted_mag210x_patch224_fp/feats_pt --split_dir multiscale\210_patient\FA_PT_k=4 --proto_path splits/multiscale/210_patient/FA_PT_k=4/prototype --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/2_10_patient_ABMIL --emb_model_type LinearEmb --sample_col case_id

## patient-level bags
C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\caseid_splits\FA_PT_k=0

python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_pt --split_dir caseid_splits\FA_PT_k=4 --split_names train --in_dim 1024 --n_proto_patches 1000000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10 --sample_col case_id

python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type allcat --data_source C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_pt --split_dir caseid_splits\FA_PT_k=4 --proto_path splits/caseid_splits/FA_PT_k=4/prototypes --in_dim 1024 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code patient_split/2.5x_allcat --emb_model_type LinearEmb --sample_col case_id
