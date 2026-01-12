# run_jobs.py
import subprocess, shlex, os, sys, datetime
from pathlib import Path

# Put each CLI as a single string (easy to edit/copy from terminal)
COMMANDS = [
    # prototype generation
    r"""python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=0 --split_names train --in_dim 2048 --n_proto_patches 400000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10 --sample_col case_id""",
    r"""python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=1 --split_names train --in_dim 2048 --n_proto_patches 400000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10 --sample_col case_id""",
    r"""python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=2 --split_names train --in_dim 2048 --n_proto_patches 400000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10 --sample_col case_id""",
    r"""python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=3 --split_names train --in_dim 2048 --n_proto_patches 400000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10 --sample_col case_id""",
    r"""python -m training.main_prototype --mode kmeans --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=4 --split_names train --in_dim 2048 --n_proto_patches 400000 --n_proto 16 --n_init 5 --seed 1 --num_workers 10 --sample_col case_id""",
    
    # training jobs
    # weight avg mean
    r"""python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=0 --proto_path splits/multiscale/52_norm_patient/FA_PT_k=0/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_norm_patient_avgmean --emb_model_type LinearEmb --sample_col case_id""",
    r"""python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=1 --proto_path splits/multiscale/52_norm_patient/FA_PT_k=1/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_norm_patient_avgmean --emb_model_type LinearEmb --sample_col case_id""",
    r"""python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=2 --proto_path splits/multiscale/52_norm_patient/FA_PT_k=2/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_norm_patient_avgmean --emb_model_type LinearEmb --sample_col case_id""",
    r"""python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=3 --proto_path splits/multiscale/52_norm_patient/FA_PT_k=3/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_norm_patient_avgmean --emb_model_type LinearEmb --sample_col case_id""",
    r"""python -m training.main_classification --task fa_vs_pt --model_type PANTHER --model_config PANTHER_fa_pt --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_norm_patient/FA_PT_k=4 --proto_path splits/multiscale/52_norm_patient/FA_PT_k=4/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_norm_patient_avgmean --emb_model_type LinearEmb --sample_col case_id""",

    # ABMIL baseline
    # r"""python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_patient/FA_PT_k=0 --proto_path splits/multiscale/52_patient/FA_PT_k=0/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_patient_ABMIL --emb_model_type LinearEmb --sample_col case_id""",
    # r"""python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_patient/FA_PT_k=1 --proto_path splits/multiscale/52_patient/FA_PT_k=1/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_patient_ABMIL --emb_model_type LinearEmb --sample_col case_id""",
    # r"""python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_patient/FA_PT_k=2 --proto_path splits/multiscale/52_patient/FA_PT_k=2/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_patient_ABMIL --emb_model_type LinearEmb --sample_col case_id""",
    # r"""python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_patient/FA_PT_k=3 --proto_path splits/multiscale/52_patient/FA_PT_k=3/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_patient_ABMIL --emb_model_type LinearEmb --sample_col case_id""",
    # r"""python -m training.main_classification --task fa_vs_pt --model_type ABMIL --model_config ABMIL_default --out_type weight_avg_mean --data_source C:/Users/Vivian/Documents/PANTHER/PANTHER/features/multiscale/52/uniextracted_mag52x_patch224_fp/feats_pt --split_dir multiscale/52_patient/FA_PT_k=4 --proto_path splits/multiscale/52_patient/FA_PT_k=4/prototypes --in_dim 2048 --n_proto 16 --batch_size 1 --max_epochs 50 --lr 5e-4 --seed 1 --exp_code multiscale/52_patient_ABMIL --emb_model_type LinearEmb --sample_col case_id"""
]

# Settings
FAIL_FAST = True   # stop on first non-zero return code
LOG_DIR = Path("job_logs") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd_str: str, idx: int):
    print(f"\n[JOB {idx+1}/{len(COMMANDS)}] Starting:")
    print(cmd_str)
    log_path = LOG_DIR / f"job_{idx+1:02d}.log"

    # On Windows, use posix=False for correct splitting of paths with spaces
    args = shlex.split(cmd_str, posix=False)

    with open(log_path, "w", buffering=1, encoding="utf-8") as log:
        log.write(f"COMMAND: {cmd_str}\nSTART: {datetime.datetime.now()}\n\n")
        # Stream output directly to the log file
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, shell=False, env={**os.environ, "PYTHONUTF8": "1"})
        
        ret = proc.wait()
        log.write(f"\nEND: {datetime.datetime.now()}\nRETURNCODE: {ret}\n")

    print(f"[JOB {idx+1}] Finished with code {ret}. Log: {log_path}")
    return ret

def main():
    for i, cmd in enumerate(COMMANDS):
        ret = run_cmd(cmd, i)
        if ret != 0 and FAIL_FAST:
            print(f"\nAborting because job {i+1} failed with code {ret}.")
            sys.exit(ret)
    print("\nAll jobs completed.")

if __name__ == "__main__":
    # Tip: run this inside your conda/venv where training deps are installed
    main()
