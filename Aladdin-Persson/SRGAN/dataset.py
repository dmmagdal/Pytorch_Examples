# dataset.py

# Windows/MacOS/Linux
# Python 3.7


import os
import config
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class MyImageFolder(Dataset):
	def __init__(self, root_dir):
		super(MyImageFolder, self).__init__()

		self.data = []
		self.root_dir = root_dir
		self.class_names = os.listdir(root_dir)

		for index, name in enumerate(self.class_names):
			files = os.listdir(os.path.join(self.root_dir, name))
			self.data += list(zip(files, [index] * len(files)))


	def __len__(self):
		return len(self.data)


	def __getitem__(self, index):
		img_file, label = self.data[index]
		root_and_dir = os.path.join(
			self.root_dir, self.class_names[label]
		)

		image = np.array(Image.open(os.path.join(
			root_and_dir, img_file
		)))
		image = config.both_transform(image)
		high_res = config.highres_transform(image)
		low_res = config.lowres_transform(image)
		return low_res, high_res


def test():
	datasest = MyImageFolder(root_dir="new_data/")
	loader = DataLoader(dataset, batch_size=1, num_workers=8)

	for low_res, high_res in loader:
		print(low_res.shape)
		print(high_res.shape)


test()