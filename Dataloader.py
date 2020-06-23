import os
import sys
import random
import torch
from random import shuffle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from enum import Enum

#Defining train, test, validaiton
class Data(Enum):
	TRAIN = 1
	TEST = 2
	VAL = 3

class MyDataset(Dataset):
	'''
	-----Data structure-----
	./trainer.py
	./Dataloader.py
	./train.txt
	./val.txt
	./test.txt

	./data/Train/class1/imagefile
				/class2
				   :
				   :
	      /Test/class1/imagefile
		           :
				   :
		  /Val/class1/imagefile
 		           :
 				   :

	class1, class2...etc -> taking as label
	'''
	def __init__(self, split = Data.TRAIN, transform = None):
		self.trainfiles = []
		self.valfiles = []
		self.testfiles = []
		self.imageFormats = ['.jpg', '.png', '.bmp', 'jpeg']
		self.split = split
		self.rootDir = 'Train'

		'''#When dataloading from directory
		for root, dirs, files in os.walk(self.rootDir):
			for file in files:
				for imageFormat in self.imageFormats:
					if file.endswith(imageFormat):
						self.trainfiles.append(os.path.abspath(os.path.join(root, file)))
						break
		'''

		# When dataloading from textfiles
		with open('train.txt', 'r') as trainfile:
			self.trainfiles = trainfile.readlines()
		with open('test.txt', 'r') as testfile:
			self.testfiles = testfile.readlines()
		with open('val.txt', 'r') as valfile:
			self.valfiles = valfile.readlines()

		# Assign IDs to the class names in the dictionary
		self.class_names = [cls for cls in os.listdir(self.rootDir) if cls != './']
		self.class_dict = {}
		for idx, cls in enumerate(self.class_names):
			self.class_dict[cls] = idx

		# Removing \n
		for idx in range(len(self.trainfiles)):
			self.trainfiles[idx] = self.trainfiles[idx].strip()
		for idx in range(len(self.testfiles)):
			self.testfiles[idx] = self.testfiles[idx].strip()
		for idx in range(len(self.valfiles)):
			self.valfiles[idx] = self.valfiles[idx].strip()

		random.seed(0) #fixing random seed
		# If you need to shuffle data, please use it
		shuffle(self.trainfiles)
		shuffle(self.testfiles)
		shuffle(self.valfiles)

		self.transform = transform

	def getNumClasses(self):
		return len(self.class_names) #it is optional value

	def __len__(self): #the number of all files
		if self.split == Data.TRAIN:
			return len(self.trainfiles)
		elif self.split == Data.TEST:
			return len(self.testfiles)
		elif self.split == Data.VAL:
			return len(self.valfiles)

	def __getitem__(self, idx):
		if self.split == Data.TRAIN:
			file = self.trainfiles[idx]
			img = Image.open(file) #loading images
			if self.transform:
				img = self.transform(img)
			#labeling
			label = self.class_dict[file.split(os.sep)[-2]]

			sample = {'data': img, 'label': label}
		elif self.split == Data.TEST:
			file = self.testfiles[idx]
			img = Image.open(file)
			if self.transform:
				img = self.transform(img)
			#labeling
			label = self.class_dict[file.split(os.sep)[-2]]

			sample = {'data': img, 'label': label}
		elif self.split == Data.VAL:
			file = self.valfiles[idx]
			img = Image.open(file)
			if self.transform:
				img = self.transform(img)
			#labeling
			label = self.class_dict[file.split(os.sep)[-2]]

			sample = {'data': img, 'label': label}

		return sample

if __name__ == '__main__':
	print('Test if dataloading is working correctly or not')

	dataTransform = transforms.Compose([
		transforms.RandomResizedCrop(224, scale = (0.7, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

	dataset = MyDataset('data', split = Data.TRAIN, transform = dataTransform)
	dataLoader = DataLoader(dataset = dataset, num_workers = 0, batch_size = 5, shuffle = False)

	for idx, data in enumerate(dataLoader):
		print('Start loading')

		X = data['data']
		y = data['label']

		print(X.shape)
		print(y.shape)

		if idx == 1:
			break
