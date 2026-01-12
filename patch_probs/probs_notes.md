# save_patch_prob.py
- script to save qq probs per slide
- optionally, set num_iters=0 in map_em(self, data, mask=None, num_iters=3, tau=1.0, prior=None) in networks.py
- ^ this shouldn't change qq values ^

*2.5x*
python save_patch_prob.py --h5_root C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_h5 --out_root C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\cross-val\2.5x\k=0 --splits train,val,test --in_dim 1024 --n_proto 16 --proto_path C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val\FA_PT_k=0\prototypes\prototypes_c16_uniextracted_mag2x_patch224_fp_kmeans_num_1.0e+06.pkl --config_dir r'C:\Users\Vivian\Documents\PANTHER\PANTHER\src\configs' --model_config PANTHER_fa_pt --out_type allcat
- ^^ old script command ^^

python -m visualization.save_patch_prob --h5_root C:/Users/Vivian/Documents/CLAM/CLAM/FEATURES_DIR_5x/FEATURES_DIR_2.5x/uniextracted_mag2x_patch224_fp/feats_h5 --split_dir C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val\FA_PT_k=4 --out_root C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\cross-val\2.5x\k=4 --in_dim 1024 --n_proto 16 --proto_path C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val\FA_PT_k=4\prototypes\prototypes_c16_uniextracted_mag2x_patch224_fp_kmeans_num_1.0e+06.pkl --config_dir C:\Users\Vivian\Documents\PANTHER\PANTHER\src\configs --model_config PANTHER_fa_pt --out_type allcat --h5_exact_names


*- update h5_root, out_root, proto_path*

*5x no norm*
python -m visualization.save_patch_prob --h5_root C:\Users\Vivian\Documents\PANTHER\PANTHER\features\unifiltextracted_mag5x_patch224_fp\feats_h5 --out_root C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\5x_filt --split_dir C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT_k=0 --splits train,val,test --in_dim 1024 --n_proto 16 --proto_path C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT_k=0\prototypes\prototypes_c16_uni_kmeans_num_1.0e+06.pkl --config_dir C:\Users\Vivian\Documents\PANTHER\PANTHER\src\configs --model_config PANTHER_fa_pt --out_type allcat

# inspect_saved_probs.py
python -m visualization.inspect_saved_probs --dir C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\5x\val --out_csv C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\5x\val\qq_npz_summary.csv

# classify_patch_probs.ipynb (start with this)
- first script created to classify qq soft assignments. 
- tested LR and MLP classifiers, used only majority and soft voting

# concat_probs_emb.ipynb 
- merged patch embeddings and qq. Tested out pytorch MLP and used attention pooling

# classify_qq_emb.py (USE THIS)
- revised script to classify qq (and optionally + emb). implement differnt aggregation methods for slide-level
classification. plots train/val loss curves. saves metrics


# classify_crossval.py
- new script to run cross val for patch-level classification

- example commands:

- navigate into PANTHER dir and out of 'src':
python patch_probs\classify_crossval.py --npz_root_tpl "C:/Users/Vivian/Documents/PANTHER/PANTHER/patch_probs/cross-val/{mag}/k={k}" --mag 2.5x --split_dir_tpl "C:/Users/Vivian/Documents/PANTHER/PANTHER/src/splits/cross-val/FA_PT_k={k}" --k 5 --label_map "FA:0,PT:1" --renorm_rows --vote_mode majority --max_iter 2000 --solver liblinear --random_state 0 --out_dir C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\cross-val\2.5x\results


qq-only
========
python patch_probs\classify_qq_emb.py --merged-root "C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\merged_filt_2.5x" --label-csv "C:\Users\Vivian\Documents\PANTHER\PANTHER\src\visualization\slides_list.csv" --save-dir "C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\merged_filt_2.5x\results" --pool attn --head lr


PCA(Z)+qq
==========
python patch_probs\classify_qq_emb.py --merged-root "C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\merged_filt_2.5x" --label-csv "C:\Users\Vivian\Documents\PANTHER\PANTHER\src\visualization\slides_list.csv" --save-dir "C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\merged_filt_2.5x\results" --no-use-only-q --pca-dim 32 --epochs 80 --batch-size-slides 6 --lr 5e-4 --weight-decay 1e-4 --patience 12 --num-workers 4 --pool mean --head mlp3

no plots and no class weights
==============================
python patch_probs\classify_qq_emb.py --merged-root "C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\merged_2.5x" --label-csv "C:\Users\Vivian\Documents\PANTHER\PANTHER\src\visualization\slides_list.csv" --save-dir "C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\merged_2.5x\results" --no-plots --no-use-class-weights

z only
=======
python patch_probs\classify_qq_emb.py --merged-root "C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\merged_filt_2.5x" --label-csv "C:\Users\Vivian\Documents\PANTHER\PANTHER\src\visualization\slides_list.csv" --save-dir "C:\Users\Vivian\Documents\PANTHER\PANTHER\patch_probs\merged_filt_2.5x\results" --feature-mode z --pca-dim 64 --epochs 80 --batch-size-slides 6 --lr 5e-4 --weight-decay 1e-4 --patience 12 --num-workers 4 --pool attn --head mlp3