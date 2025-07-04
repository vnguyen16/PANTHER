import pandas as pd
import os

def normalize_id(name):
    return name.strip().replace(" ", "")  # Remove extra spaces like in "FA 60 B"

def create_panther_splits(
    full_csv_path,
    old_train_path,
    old_val_path,
    old_test_path,
    output_dir
):
    # Load the full PANTHER-format CSV
    full_df = pd.read_csv(full_csv_path)
    full_df['slide_id_norm'] = full_df['slide_id'].apply(normalize_id)

    # Helper to process a single old-format split file
    def load_old_split(path):
        df = pd.read_csv(path)
        df['Filename_norm'] = df['Filename'].apply(normalize_id)
        return df['Filename_norm'].tolist()

    # Load each split
    train_ids = load_old_split(old_train_path)
    val_ids   = load_old_split(old_val_path)
    test_ids  = load_old_split(old_test_path)

    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Match and save new splits
    for split_name, ids in zip(['train', 'val', 'test'], [train_ids, val_ids, test_ids]):
        split_df = full_df[full_df['slide_id_norm'].isin(ids)].copy()
        split_df = split_df[['slide_id', 'label']]  # Drop helper column
        split_df.to_csv(os.path.join(output_dir, f'{split_name}.csv'), index=False)
        print(f"{split_name}: {len(split_df)} slides written")


def convert_labels_to_strings(split_dir, label_map={0: 'FA', 1: 'PT'}, overwrite=True):
    """
    Converts numeric labels (e.g., 0, 1) in train/val/test split CSVs to string labels (e.g., 'FA', 'PT').

    Parameters:
        split_dir (str): Path to the directory containing the split CSVs.
        label_map (dict): Mapping from numeric to string labels.
        overwrite (bool): If True, overwrites the original CSVs. Otherwise, saves as new files.
    """
    splits = ['train.csv', 'val.csv', 'test.csv']
    for split in splits:
        csv_path = os.path.join(split_dir, split)
        df = pd.read_csv(csv_path)
        
        # Replace numeric label with string label
        df['label'] = df['label'].map(label_map)

        if overwrite:
            df.to_csv(csv_path, index=False)
            print(f"✅ Overwrote {csv_path} with string labels.")
        else:
            new_path = os.path.join(split_dir, f"string_{split}")
            df.to_csv(new_path, index=False)
            print(f"✅ Saved converted labels to {new_path}")


if __name__ == "__main__":
    # create_panther_splits(
    #     full_csv_path=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT\fa_vs_pt.csv",
    #     old_train_path=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT\train_split.csv",
    #     old_val_path=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT\val_split.csv",
    #     old_test_path=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT\test_split.csv",
    #     output_dir="src/splits/FA_PT"
    # )

    convert_labels_to_strings(r'C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\FA_PT_k=0')

