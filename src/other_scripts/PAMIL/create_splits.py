# write a script that takes in 3 separate csv files (train, val, test)
# and combines them into a single csv file with an additional column
# indicating the split (train, val, test)

import pandas as pd

# File paths (update these to match your actual filenames)
train_file = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val\FA_PT_k=4\train.csv"
val_file = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val\FA_PT_k=4\val.csv"
test_file = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val\FA_PT_k=4\test.csv"
output_file = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\other_scripts\PAMIL\crossval_splits\splits_4.csv" # save as 'splits_{fold}.csv

# Read the CSVs
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

# Extract only slide_id column
train_ids = train_df["slide_id"]
val_ids = val_df["slide_id"]
test_ids = test_df["slide_id"]

# Combine into one dataframe, aligning by index (fill shorter columns with NaN)
combined_df = pd.DataFrame({
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
})

# Save to new CSV
combined_df.to_csv(output_file, index=False)

print(f"Combined file saved to {output_file}")
