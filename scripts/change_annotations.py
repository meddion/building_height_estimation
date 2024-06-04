import json
import glob
import os
from dataset import get_height_class, NUMBER_OF_CLASSES

from pathlib import Path

output_dir = Path("datasets/mlc_training_data/ground_truth_files_class")
os.makedirs(output_dir, exist_ok=True)

all_labels = [
    "__ignore__",
    "_background_",
]

for i in range(1, NUMBER_OF_CLASSES):
    all_labels.append("building_class_" + str(i))

with open('labels_new.txt', 'w') as f:
    for label in all_labels:
        f.write(label+"\n")

files = glob.glob("datasets/mlc_training_data/ground_truth_files/*.json")
for i, file_name in enumerate(files):
    print(f"Processing: {file_name} ({i+1}/{len(files)})")
    with open(file_name) as f:
        annot = json.load(f)
        for i in range(len(annot["shapes"])):
            annot["shapes"][i]["label"] = "building_class_"+ str(get_height_class(annot["shapes"][i]["group_id"]))

        json.dump(annot, open(output_dir / Path(file_name).name, "w"), indent=4)
        print(f"Processed: {file_name} ({i+1}/{len(files)})")
