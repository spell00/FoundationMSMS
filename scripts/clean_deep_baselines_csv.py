import csv

# Input and output paths
input_path = "logs/deep_baselines/baselines_fold_metrics.csv"
output_path = "logs/deep_baselines/baselines_fold_metrics.cleaned.csv"

header = [
    "dataset","model","fold","train_size","val_size","test_size","n_classes",
    "train_accuracy","train_balanced_accuracy","train_macro_f1","train_mcc",
    "val_accuracy","val_balanced_accuracy","val_macro_f1","val_mcc"
]

with open(input_path, newline='', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for i, row in enumerate(reader):
        if i == 0:
            writer.writerow(header)
        else:
            # Always take the last 15 columns
            if len(row) >= 15:
                writer.writerow(row[-15:])

print(f"Cleaned file written to {output_path}")
