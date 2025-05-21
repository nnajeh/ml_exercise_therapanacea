from  bib import *
from models import *
from data.data_loader import get_data_loaders



# path to train and valid datasets
train_img_dir = './ml_exercise_therapanacea/train_img/'
val_img_dir = './ml_exercise_therapanacea/val_img/'
train_labels_file = './ml_exercise_therapanacea/label_train.txt'


if __name__ == '__main__':

    # path to train and valid datasets
    train_img_dir = './ml_exercise_therapanacea/train_img/'
    val_img_dir = './ml_exercise_therapanacea/val_img/'
    train_labels_file = './ml_exercise_therapanacea/label_train.txt'

    
    train_loader, val_loader = get_data_loaders(train_img_dir, train_labels_file, val_img_dir)

    train_model(train_loader)
  
    predict(val_loader)
