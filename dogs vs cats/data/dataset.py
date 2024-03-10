import os
from PIL import Image
from torch.utils import data
from torchvision import transforms

class DogCat(data.Dataset):
    def __init__(self,root,transform=None,test=False,train=True):
        """
        主要目的：划分数据
        root:数据集的路径
        """
        self.test = test
        imgs = [os.path.join(root,img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        # 根据不同的数据集划分imgs
        if self.test:
            imgs = sorted(imgs,key=lambda x: int(x.split('.')[-2].split('\\')[-1]))
        else:
            imgs = sorted(imgs,key=lambda x: int(x.split('.')[-2]))

        #划分训练集，验证集和测试集
        imgs_len = len(imgs)
        #训练集和验证集之比为 3 : 7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[0:int(0.7*imgs_len)]
        else:
            self.imgs = imgs[int(0.7*imgs_len):]
        #对数据进行增强
        if transform is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
            if self.test:
                self.transform = transforms.Compose([transforms.Resize(224),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    normalize])
            else:
                self.transform = transforms.Compose(
                                                    [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                                    transforms.RandomRotation(degrees=15),
                                                    transforms.ColorJitter(),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize])
    #图片读取等耗时的操作放在
    def __getitem__(self, index):
        """
        :param index:图片位置的索引
        :return:返回一张图片的数据，对于测试集，没有label，返回图片id
        """
        img_path = self.imgs[index]

        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('\\')[-1])
        else:
            label = 1 if 'dog' in self.imgs[index].split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transform(data)

        return data,label


    def __len__(self):
        return len(self.imgs)









