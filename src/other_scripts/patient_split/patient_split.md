- created split csv with added caseid column --> located in C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\caseid_splits

- added sample_col arg to *main_classification.py*
- added sample col to train_kwargs and val_kwargs

- added arg to *main_embedding.py*
- added arg mean train
def main(args):
    
    train_kwargs = dict(data_source=args.data_source, sample_col=args.sample_col) # added sample_col arg

- added arg to *main_prototype.py*