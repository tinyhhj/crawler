from PIL import Image
import os
import torch
import shutil

def image_check(root):
    for file in os.listdir(root):
        if not os.path.isdir(os.path.join(root,file)):
            try:
                Image.open(os.path.join(root,file))
            except:
                print('removed {}'.format(os.path.join(root, file)))
                os.remove(os.path.join(root,file))


def random_split(root):
    files = os.listdir(root).copy()
    for f in os.listdir(root):
        if os.path.isdir(os.path.join(root,f)):
            files.remove(f)
    train_num = len(files) * 7 // 10
    val_num = len(files) * 15 // 100
    test_num = len(files) -train_num - val_num
    print('train val test split: {} {} {}'.format(train_num, val_num, test_num))
    return torch.utils.data.random_split(files,[train_num, val_num, test_num])

def move_files(src,dest,train, val, test):
    for f in train:
        shutil.move(os.path.join(src,f),os.path.join(dest,'train\\class',f))
    for f in val:
        shutil.move(os.path.join(src,f), os.path.join(dest,'val\\class',f))
    for f in test:
        shutil.move(os.path.join(src,f),os.path.join(dest,'test\\class',f))

# image_check('zigbang')
# datasets = random_split('zigbang')
# move_files('zigbang', 'zigbang',*datasets)
# image_check('zigbang/train/class')
# image_check('zigbang/test/class')
# image_check('zigbang/val/class')


