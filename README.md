# Image Classification Exercise

## Objective
Develop an image classifier that minimizes the Half-Total Error Rate (HTER)

## Dataset
- Training images: 100,000 in train_img/
- Training labels: label_train.txt (binary: 0 or 1)
- Validation images: 20,000 in val_img/

## Deliverables
1. label_val.txt — 20k binary predictions.
2. Code — Jupyter notebook or Python scripts, well-commented and structured.

## Project Structure
- data/: Data loading and preprocessing
- models/: Training and evaluation scripts
- exercise.ipynb: Jupyter notebook for training
- output/: Output files like label_val.txt

## Requirements
```bash (conda environment)
pip install torch torchvision numpy pandas scikit-learn matplotlib
