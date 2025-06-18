import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset


# 定义类别到标签的映射
class_to_label = {
    'group00': 0,
    'group02': 1,
    'group04': 2,
    'group06': 3,
    'group08': 4,
    'group10': 5,
    'group12': 6,
    'group14': 7,
}

# # 定义类别到标签的映射(2点一组)
# class_to_label = {
#     'group1':  0,
#     'group2':  0,
#     'group4':  1,
#     'group5':  1,
#     'group7':  2,
#     'group8':  2,
#     'group10': 3,
#     'group11': 3,
#     'group13': 4,
#     'group14': 4,
# }

# 定义类别到标签的映射
class_to_label_test = {
    'group00': 0,
    'group01': 1,
    'group02': 2,
    'group03': 3,
    'group04': 4,
    'group05': 5,
    'group06': 6,
    'group07': 7,
    'group08': 8,
    'group09': 9,
    'group10': 10,
    'group11': 11,
    'group12': 12,
    'group13': 13,
    'group14': 14,
    'group15': 15,
}



def get_label(img_path, test = True):
    # 提取文件名 train: data/train/group1.123.jpg 
    filename = img_path.split('/')[-1].split('.')[0]
    
    # 遍历类别映射，找到匹配的类别
    if test:
        for class_name, label in class_to_label_test.items():
            if class_name == filename:
                return label
    else :
        for class_name, label in class_to_label.items():
            if class_name == filename:
                return label
    
    # 如果没有匹配的类别，可以选择返回默认值或处理异常
    return -1

class IndoorGroup(Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        if self.test:
            # test1: data\test\group1.1.jpg
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        else:
            # train: data\train\group0.11.jpg
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)

        #划分数据集
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.8 * imgs_num)]
        else:
            self.imgs = imgs[int(0.8 * imgs_num):]

        if transforms is None:
            if self.test or not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ])
            else:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        if self.test:
            # test1: data\test\group1.45.jpg
            label = get_label(img_path, test=True)
        else:
            # train: data\train\group0.38.jpg
            label = get_label(img_path, test=False)
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
    


if __name__ == "__main__":
    #准备数据集
    train_dataset = IndoorGroup('./data/train/', train=True)
    val_dataset = IndoorGroup('./data/train/', train=False)
    test_data = IndoorGroup('./data/test_3/',test=True)

    #利用DataLodear来加载数据集
    batch_size = 64
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=7)
    test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle=False, num_workers=7)

    print(len(train_dataset))
    for ii, (data, label) in enumerate(trainloader):
        if ii % 100 == 0:
            print(data.shape, label)
    print(len(val_dataset))
    for ii, (data, label) in enumerate(valloader):
        if ii % 100 == 0:
            print(data.shape, label)
    print(len(test_data))
    for ii, (data, label) in enumerate(test_dataloader):
        if ii % 100 == 0:
            print(data.shape, label)