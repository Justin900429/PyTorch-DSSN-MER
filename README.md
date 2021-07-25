# Pytorch DSSN-MER README

## Introduction
This is the Pytorch Version for the reproduction of [Dual-Dtream Shallow Networks for Facial Micro-Expression Recognition](https://ieeexplore.ieee.org/document/8802965).

## Files
* **dataloader**: Function for building custom dataset
* **network**: Network design
* **preprocess**: Function for processing the original dataset
* **save_preprocess_weight**: Save the preprocessed image into npz for repeating usage
* **train**: Train the model
* **utils**: Rename the fileanem in SAMM dataset

## Package requirements
Install the required packages in this project by:
```shell
$ pip install -r requirements.txt
```

## Usage
> Note that for SAMM dataset, you should rename the data to fulfill the format in the code. To rename the dataset, see `utils.py`.
> 
### Preprocessing
The preprocessed images can be reused to cut down the training time. The usage is shown below:
```
usage: save_process_image.py [-h] --csv_path CSV_PATH --root ROOT --save_path
                             SAVE_PATH [--CASME]

optional arguments:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH
  --root ROOT
  --save_path SAVE_PATH
  --CASME
```

For example:
```shell
$ python save_process_weight \
    --csv_path <path_to_csv_file> \
    --root <Path to images> \
    --save_path CASME_preprocess \
    --CASME
```

### Training
```
usage: train.py [-h] --path PATH --pre PRE --catego CATEGO
                [--num_classes NUM_CLASSES]
                [--combination_mode COMBINATION_MODE]
                [--image_mode IMAGE_MODE [IMAGE_MODE ...]]
                [--batch_size BATCH_SIZE]
                [--weight_save_path WEIGHT_SAVE_PATH] [--model MODEL]
                [--freeze_k FREEZE_K] [--epochs EPOCHS]
                [--learning_rate LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path for the csv file for training data
  --pre PRE             Preprocess path for training
  --catego CATEGO       SAMM or CASME dataset
  --num_classes NUM_CLASSES
                        Classes to be trained
  --combination_mode COMBINATION_MODE
                        Mode to be used in combination
  --image_mode IMAGE_MODE [IMAGE_MODE ...]
                        Image type to be used in training
  --batch_size BATCH_SIZE
                        Training batch size
  --weight_save_path WEIGHT_SAVE_PATH
                        Path for the saving weight
  --model MODEL         Model to used for training
  --freeze_k FREEZE_K   Layer to freeze in AlexNet
  --epochs EPOCHS       Epochs for training the model
  --learning_rate LEARNING_RATE
                        Learning rate for training the model
```

For example:
```shell
$ python train.py \
    --path <path to csv file> \
    --pre CASME_preprocess \
    --catego CASME \
    --image_mode F G \
    --weight_save_path CASME_DSSN_weight \
    --model DSSN
```

Especially, user can decide what mode to be used in training. **DSSN** need two modes for training. On the other hand, **SSSN** only need to choose one mode.
```
# For DSSN
... \
--image_mode F G \
...

# For SSSN
... \
--image_mode G
```

## Pretrained AlexNet
The weight was obtained from [YOUSIKI's](https://github.com/YOUSIKI/PyTorch-AlexNet) GitHub repo.

## Metrics

### Model weights


| Model | Paramters |
| ----- | --------- |
| SSSN  | 0.49M     |
| DSSN  | 0.82M     |

### Evaluation (LOSO cross validation)
|  Categories  | Accuracy | F1-Score |          Method           |
|:------------:|:--------:|:--------:|:-------------------------:|
| SAMM (DSSN)  |  0.5664  |  0.5933  | "Magnitude + Gray", "Add" |
| SAMM (SSSN)  |  0.5551  |  0.5647  |          "Gray"           |
| CASME (DSSN) |  0.6432  |  0.6378  | "Magnitude + Gray", "Add" |
| CASME (SSSN) |  0.6249  |  0.6289  |          "Gray"           |

> F1-score was computed using **weighted** method


## Citation
```bibtex
@inproceedings{khor2019dual,
    title={Dual-stream Shallow Networks for Facial Micro-expression Recognition},
    author={Khor, Huai-Qian and See, John and Liong, Sze-Teng and Phan, Raphael CW and Lin, Weiyao},
    booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
    pages={36--40},
    year={2019},
    organization={IEEE}
}
```
