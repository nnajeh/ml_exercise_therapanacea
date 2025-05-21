from bib import *

class ImageDataset(Dataset):
    def __init__(self, img_dir, labels_file=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted(os.listdir(img_dir))
        
        if labels_file:
            self.labels = pd.read_csv(labels_file, header=None, names=['label'])
        else:
            self.labels = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels.iloc[idx, 0]
            return image, label
        else:
            return image

        
def get_data_loaders(train_img_dir, train_labels_file, val_img_dir, batch_size=256):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageDataset(train_img_dir, train_labels_file, transform=transform)
    val_dataset = ImageDataset(val_img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader
