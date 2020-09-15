# --------------------------------------------------------
# Adaptation of FDA to Intra DA
#
#
# Updated by Hojun Lim
# Update date 12.09.2020
# --------------------------------------------------------



from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data


class BaseDataset(data.Dataset):
    def __init__(self, args, root, list_path, set_,
                 max_iters, image_size, labels_size, mean):
        self.args = args
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        print(self.list_path)
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file, label_file = self.get_metadata(name)
            self.files.append((img_file, label_file, name))

    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        """
        self.set indicates type of dataset. 
        'all' is a keyword to denote gta5 dataset(as a source domain, there is no distinction between train set and validation set, but the whole data is used for training)
        'train' is a keyword for training set of Cityscape dataset, which is used used only for training.
        'val' is a keyword for validation set of Cityscape dataset.
        """
        # Preprocessing step in evaluation
        if self.set == 'val':
            image -= self.mean

        # preprocessing step in training
        elif self.set == 'train' or self.set == 'all':
            if self.args.FDA_mode == 'off':
                image -= self.mean
            elif self.args.FDA_mode == 'on':
                pass # subtraction by mean from image will be conducted after the amplitude switch(FDA)
            else:

                raise KeyError()
        else:

            raise KeyError()

        return image.transpose((2, 0, 1))

    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return _load_img(file, self.labels_size, Image.NEAREST, rgb=False)


def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)
