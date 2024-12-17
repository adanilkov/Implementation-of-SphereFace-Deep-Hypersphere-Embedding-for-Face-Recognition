import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class LFW4Training(Dataset):
    def __init__(self, train_file: str, img_folder: str):
        self.img_folder = img_folder

        names = os.listdir(img_folder)
        self.name2label = {name: idx for idx, name in enumerate(names)}
        self.n_label = len(self.name2label)

        with open(train_file) as f:
            train_meta_info = f.read().splitlines()

        self.train_list = []
        for line in train_meta_info:
            line = line.split("\t")
            if len(line) == 3:
                self.train_list.append(os.path.join(line[0], line[0] + "_" + str(line[1]).zfill(4) + ".jpg"))
                self.train_list.append(os.path.join(line[0], line[0] + "_" + str(line[2]).zfill(4) + ".jpg"))
            elif len(line) == 4:
                self.train_list.append(os.path.join(line[0], line[0] + "_" + str(line[1]).zfill(4) + ".jpg"))
                self.train_list.append(os.path.join(line[2], line[2] + "_" + str(line[3]).zfill(4) + ".jpg"))
            else:
                pass

        self.transform = transforms.Compose([
            transforms.Resize(96),
            transforms.RandomRotation(degrees=10),  # Random rotation up to 10 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        img_path = self.train_list[index]

        img = Image.open(os.path.join(self.img_folder, img_path))
        img = self.transform(img)

        name = img_path.split("/")[0]
        label = self.name2label[name]

        return img, label

    def __len__(self):
        return len(self.train_list)


class LFW4Eval(Dataset):
    def __init__(self, eval_file: str, img_folder: str):
        self.img_folder = img_folder

        with open(eval_file) as f:
            eval_meta_info = f.read().splitlines()

        self.eval_list = []
        for line in eval_meta_info:
            line = line.split("\t")
            if len(line) == 3:
                eval_pair = (
                    os.path.join(line[0], line[0] + "_" + str(line[1]).zfill(4) + ".jpg"),
                    os.path.join(line[0], line[0] + "_" + str(line[2]).zfill(4) + ".jpg"),
                    1,
                )
                self.eval_list.append(eval_pair)
            elif len(line) == 4:
                eval_pair = (
                    os.path.join(line[0], line[0] + "_" + str(line[1]).zfill(4) + ".jpg"),
                    os.path.join(line[2], line[2] + "_" + str(line[3]).zfill(4) + ".jpg"),
                    0,
                )
                self.eval_list.append(eval_pair)
            else:
                pass

        self.transform = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        img_1_path, img_2_path, label = self.eval_list[index]

        img_1 = Image.open(os.path.join(self.img_folder, img_1_path))
        img_2 = Image.open(os.path.join(self.img_folder, img_2_path))
        img_1 = self.transform(img_1)
        img_2 = self.transform(img_2)

        return img_1, img_2, label

    def __len__(self):
        return len(self.eval_list)

