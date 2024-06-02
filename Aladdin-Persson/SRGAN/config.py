import torch
from PIL import Image
from torchvision.transforms as transforms


LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.path.tar"
CHECKPOINT_DISC = "disc.path.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 4
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

highres_transform = transforms.Compose(
	[
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
	]
)

lowres_transform = transforms.Compose(
	[
		transforms.ToTensor(),
		transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
		transforms.RandomCrop(size=(LOW_RES,LOW_RES)),
	]
)

both_transform = transforms.Compose(
	[
		transforms.RandomRotation(degrees=90),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomCrop(size=(HIGH_RES, HIGH_RES)),
	]
)

test_transform = transforms.Compose(
	[
		transforms.ToTensor(),
		transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
	]
)