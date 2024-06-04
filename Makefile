gen-uniform-masks:
	python scripts/labelme2voc.py --labels scripts/labels.txt datasets/mlc_training_data/images_annotated_uniform datasets/mlc_training_data/masks_unifrom

gen-masks:
	python scripts/labelme2voc.py --labels scripts/labels_new.txt datasets/mlc_training_data/images_annotated datasets/mlc_training_data/masks
