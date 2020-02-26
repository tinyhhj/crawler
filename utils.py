import numpy as np
import torchvision.models as models
import torch
import os
import logging
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import colorsys


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


debug = True
logging.basicConfig(level = logging.DEBUG if debug else logging.INFO)

def boxsize(w,h):
    return w*h
def masking(device,images,names,th= 0.3, min_sat = 0.3):
    """
    random object masking from image
    :param image: tensor image (c,h,w)
    :return: randomly object masked image (h,w)
    """
    batch_size = images.size()[0]
    h = images.size()[2]
    w = images.size()[3]
    result_masks = []

    model  = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)

    model.eval()
    results = model(images.to(device))
    new_names = []

    for i,(result,name) in enumerate(zip(results,names)):

        box_args = []
        # if '17677818_0' in name:
        #     view_image(result, images[i].float())
        for i,box in enumerate(result['boxes']):
            x1, y1, x2, y2 = box
            if boxsize(x2-x1,y2-y1) <= 15000:
                box_args.append(i)
        masks = result['masks'][box_args].squeeze(1) >= th

        # nothing to detect
        if len(masks) == 0:
            logging.debug('nothing to dectect {}'.format(name))
            continue
        # UInt8Tensor[N, 1, H, W]
        # masks = result['masks'].squeeze(1) >= th
        sub_masks = masks
        #max: Returns a namedtuple (values, indices)
        if min_sat < masks.max(0)[0].sum().float() / (h*w):
            # pick random masks
            start = np.random.randint(len(masks)) + 1
            for mask_num in range(start, len(masks)+1):
                sub_masks = masks[np.random.choice(len(masks), mask_num, replace=False)]
                if min_sat < sub_masks.max(0)[0].sum().float() / (h*w):
                    break

        logging.debug('min_sat: {}, sub_masks: {}'.format(min_sat, sub_masks.max(0)[0].sum().float() / (h*w)))
        print(torch.unique((sub_masks.max(0)[0].byte()* 255).unsqueeze(0)))
        result_masks.append((sub_masks.max(0)[0].byte()* 255).unsqueeze(0))
        new_names.append(name)
    try:
        result_masks = torch.cat(result_masks)
    except:
        result_masks = []
        print(result_masks)

    return result_masks.to(torch.device('cpu')) if len(result_masks) > 0 else result_masks, new_names

def load_flist(flist):
    if os.path.isfile(flist):
        try:
            return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        except:
            return [flist]
def save_masks(output, flist, masks):
    for file, mask in zip(flist,masks):
        logging.debug('{} save'.format(os.path.join(output, os.path.split(file)[1])))
        image = Image.fromarray(mask)

        image.save(os.path.join(output, os.path.splitext(os.path.split(file)[1])[0]+'.png'))

def preprocess(device,batch):
    images = []
    names = []
    for img in batch:
        try:
            image = Image.open(img)
            if not image.size[0] == 256 or not image.size[1] == 256:
                image = image.resize((256,256))
        except:
            logging.debug('cant open {}'.format(img))
            continue
        images.append(np.transpose(np.array(image),(2,0,1)))
        names.append(img)
    images = np.array(images)
    result = torch.from_numpy(images).to(device=device).float() / 255
    print(result.size())
    return result,names
def mask(input = None, output = None,size = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='input images to mask',default = None)
    parser.add_argument('--output', type=str, help='output dir to save mask image', default= None)
    parser.add_argument('--size', type=int, help='batch size', default= 3)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")

    args = parser.parse_args()
    if not input is None:
        args.input = input
    if not output is None:
        args.output = output
    if not size is None:
        args.size = size
    os.makedirs(args.output, exist_ok=True)

    files = load_flist(args.input)
    for f in range(0,len(files),args.size):
        batch = files[f:f+args.size]
        # open이 안됐을 경우에 mask와 mapping 어그러짐
        images,names = preprocess(device,batch)
        assert len(images) == len(names)
        # mask를 dectection못했을 경우에 sum reduction 불가
        masks,names = masking(device,images,names)
        if len(masks) == 0:
            continue
        assert len(masks) == len(names)

        save_masks(args.output, names, masks.numpy())
def view_image(result,image):
    image = image.to(device=torch.device('cpu'))
    _, mp = plt.subplots()
    masks = (result['masks'].squeeze(1) >= 0.3).to(device=torch.device('cpu'))
    labels,scores = result['labels'], result['scores']
    boxes = result['boxes']
    for j in range(masks.size()[0]):
        # if( scores[j] < 0.3): continue
        print(COCO_INSTANCE_CATEGORY_NAMES[labels[j]], scores[j])
        if not torch.any(boxes[j].type(torch.uint8)): continue

        x1, y1, x2, y2 = boxes[j].type(torch.uint8)
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7, linestyle='dashed',
                              edgecolor=colors[j]
                              , facecolor='none')
        mp.add_patch(p)
        mp.text(x1, y1 + 8, COCO_INSTANCE_CATEGORY_NAMES[labels[j]], color='w', size=11, backgroundcolor='none')
        for i in range(3):
            image[i, :, :] = torch.where(masks[j], torch.tensor(255).type(torch.float), image[i, :, :])

    mp.imshow(np.transpose(image.type(torch.ByteTensor).numpy(),(1,2,0)))
    plt.show()

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
colors = random_colors(91)
class MyDataset(Dataset):
    def __init__(self,list):
        pass
    def __len__(self):
        return len(self.list)
        

if __name__ =='__main__':
    # flist = load_flist('test.flist')
    # from PIL import Image
    # t = torch.empty(0)
    # for f in flist:
    #     t= torch.cat((t,torch.from_numpy(np.transpose(np.array(Image.open(f)),(2,0,1))).to(dtype=torch.float).unsqueeze(0)))
    # r = masking(t/255).numpy()
    # import matplotlib.pyplot as plt
    # os.makedirs('mask',exist_ok=True)
    # for i,m in enumerate(r):
    #     Image.fromarray(m).save('mask/{}'.format(os.path.split(flist[i])[1]))
    mask('test.flist','mask/test', 5)