# --------------------------------------------------------
# Adaptation of FDA to Intra DA
#
#
# Written by Hojun Lim
# Update date 08.09.2020
# --------------------------------------------------------

import numpy as np

from advent.dataset.base_dataset import BaseDataset


class GTA5DataSet(BaseDataset):
    def __init__(self, args, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(args, root, list_path, set, max_iters, crop_size, None, mean)


        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def get_metadata(self, name):
        img_file = self.root / 'images' / name
        label_file = self.root / 'labels' / name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        #if self.args.FDA_mode == 'off':
        """
            if FDA_mode == on: need to conduct the amplitude switch between source and target before subtracting the image by its mean value(preprocess). 
            Therefore, perform the preprocessing only if FDA_mode == off.            
            """
        #pass
        image = self.preprocess(image)

        return image.copy(), label_copy.copy(), np.array(image.shape), name
