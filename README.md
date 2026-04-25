# BASST

Official code for **BASST** for ASD video classification from facial videos.

The original child dataset used in this work is **not public** due to privacy restrictions. To use this repository, prepare your own dataset in the format described below.

---

## What this repository supports

This repository supports two use cases:

1. **Testing / inference** on a user-provided dataset using the released pretrained checkpoint `best.pth`
2. **Fine-tuning** on a user-provided dataset starting from the released pretrained checkpoint `best.pth`

---

## Repository structure

Organize the repository like this:

```text
BASST/
├── run_folds.py
├── basst_model.py
├── auxiliary.py
├── dataset.py
├── transforms.py
├── rand_augment.py
├── random_erasing.py
├── openface3_gaze_backbone.py
├── requirements.txt
├── README.md
├── checkpoints/
│   └── best.pth
├── OpenFace-3.0/
│   └── ...
└── data/
    ├── clips/
    │   ├── sample_001/
    │   │   ├── 00001.jpg
    │   │   ├── 00002.jpg
    │   │   └── ...
    │   ├── sample_002/
    │   └── ...
    └── splits/
        ├── train_fold_1.csv
        ├── test_fold_1.csv
        ├── train_fold_2.csv
        ├── test_fold_2.csv
        └── ...
```

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## OpenFace-3.0 setup

This code requires **OpenFace-3.0**.

Download OpenFace-3.0 from:

`https://github.com/CMU-MultiComp-Lab/OpenFace-3.0`

After downloading it, place it inside this repository using the folder name:

```text
OpenFace-3.0
```

So the structure should be:

```text
BASST/
└── OpenFace-3.0/
    └── ...
```

---

## Pretrained checkpoint

Download `best.pth` from the [GitHub Releases page](https://github.com/wondimagegn-b/BASST/releases/tag/v1.0) and place it in:

```text
checkpoints/best.pth

This checkpoint can be used for both:
- **testing / inference**
- **fine-tuning** on a new dataset

---

## Dataset format

This repository expects **frame-folder input**, not raw videos.

Each sample must be stored in its own folder containing extracted RGB frames.

Example:

```text
data/clips/sample_001/
data/clips/sample_002/
...
```

Inside each sample folder:

```text
00001.jpg
00002.jpg
00003.jpg
...
```

### CSV split format

Each CSV file must contain two columns:

```text
sample_path,label
```

Example:

```text
sample_001,1
sample_002,0
sample_003,1
```

Where:
- `sample_path` is the relative path to the sample folder with respect to `data/clips`
- `label` is the class label

If you use:

```text
--data_path data/clips
```

then the row

```text
sample_001,1
```

means the code will load frames from:

```text
data/clips/sample_001/
```

### Label convention

Use the following binary labels:

- `0` = non-ASD
- `1` = ASD

If you use a different label meaning, keep it consistent across all splits.

---

## Fine-tuning

To fine-tune BASST on your own dataset using the released pretrained checkpoint:

### Windows CMD

```bat
python run_folds.py --mode train --folds 1 --data_path "data\clips" --train_label_template "data\splits\train_fold_{fold}.csv" --test_label_template "data\splits\test_fold_{fold}.csv" --output_dir "outputs" --finetune_path "checkpoints\best.pth" --openface_repo_root "OpenFace-3.0" --nb_classes 2 --input_size 224 --short_side_size 224 --num_frames 16 --sampling_rate 1 --batch_size 4 --eval_batch_size 4 --epochs 50 --lr 1e-5
```

The best checkpoint after fine-tuning will be saved as:

```text
outputs/fold_1/best.pth
```

---

## Testing / inference

To evaluate a checkpoint on your own dataset:

### Windows CMD

```bat
python run_folds.py --mode test --folds 1 --data_path "data\clips" --train_label_template "data\splits\train_fold_{fold}.csv" --test_label_template "data\splits\test_fold_{fold}.csv" --output_dir "outputs" --checkpoint_path "checkpoints\best.pth" --openface_repo_root "OpenFace-3.0" --nb_classes 2 --input_size 224 --short_side_size 224 --num_frames 16 --sampling_rate 1 --eval_batch_size 4
```

To test a checkpoint produced after fine-tuning, replace:

```text
checkpoints\best.pth
```

with:

```text
outputs\fold_1\best.pth
```

---

## Multiple folds

You can run multiple folds by listing them after `--folds`.

Example:

```bat
python run_folds.py --mode train --folds 1 2 3 --data_path "data\clips" --train_label_template "data\splits\train_fold_{fold}.csv" --test_label_template "data\splits\test_fold_{fold}.csv" --output_dir "outputs" --finetune_path "checkpoints\best.pth" --openface_repo_root "OpenFace-3.0" --nb_classes 2 --input_size 224 --short_side_size 224 --num_frames 16 --sampling_rate 1 --batch_size 4 --eval_batch_size 4 --epochs 50 --lr 1e-5
```

This will create:

```text
outputs/fold_1/
outputs/fold_2/
outputs/fold_3/
```

and:

```text
outputs/summary.json
```

---

## Output files

Typical output files include:

```text
outputs/fold_1/history.json
outputs/fold_1/test_metrics.json
outputs/fold_1/best.pth
outputs/fold_1/last.pth
outputs/summary.json
```

---

## Notes

- The original child dataset is not released for privacy reasons.
- Users must prepare their own dataset in the required frame-folder and CSV format.
- OpenFace-3.0 must be downloaded separately and placed in the repository as described above.
- In the current script interface, both `--train_label_template` and `--test_label_template` are required arguments, even in `--mode test`.

---

## Citation

If you use this code, please cite the corresponding BASST paper.

```text
[Multi-Cue Behavior-Aware Transformer for Autism Spectrum Disorder Screening from Facial Videos]
```
