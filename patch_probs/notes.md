# classify_patch_probs.ipynb 
    --> first script created to classify qq soft assignments. 
    - tested LR and MLP classifiers, used only majority and soft voting

# concat_probs_emb.ipynb 
    --> merged patch embeddings and qq. Tested out pytorch MLP and used attention pooling

#classify_qq_emb.py 
    --> revised script to classify qq (and optionally + emb). implement differnt aggregation methods for slide-level
    classification. plots train/val loss curves
