# Data preperation

1. Download training and testing data:

- [mlc_training_data.zip](https://drive.google.com/file/d/1HY4x2ZrqK9Xi2JZavxYLnJgzxz0ZSdLB/view)
- [mlc_test_images.zip](https://drive.google.com/file/d/1gxGWuMctIwlB1mFTGKYGT44kFX8L-MND/view)

2. Unzip to the `datasets` directory which is at the root of this project:

```shell
makdir -p datasets
unzip mlc_training_data.zip -d datasets/mlc_training_data
unzip mlc_test_data.zip -d datasets/mlc_training_data
```

3. Copy annotations and images to the `images_annotated` directory:

```shell
mkdir -p datasets/mlc_training_data/images_annotated
cp datasets/mlc_training_data/ground_truth_files/* datasets/mlc_training_data/images_annotated
cp datasets/mlc_training_data/images/* datasets/mlc_training_data/images_annotated
```

4. Create masks for the images:

```shell
python labelme_gen_mask_img.py
```

5. Open with labelme to verfiy:

```shell
labelme datasets/mlc_training_data/images_annotated
```

# Model

To train the model:

```shell
python model.py
```

# TODO:

- Pretrain Mask R-CNN model instance segmentation model on [Miyazaki dataset](https://ieee-dataport.org/open-access/dataset-detecting-buildings-containers-and-cranes-satellite-images).
