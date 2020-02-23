import numpy as np
import torchvision.models as models
import torch
import os
import logging
import argparse
from PIL import Image



debug = True
logging.basicConfig(level = logging.DEBUG if debug else logging.INFO)


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

        # nothing to detect
        if len(result['masks']) == 0:
            continue
        # UInt8Tensor[N, 1, H, W]
        masks = result['masks'].squeeze() >= th
        sub_masks = masks
        #max: Returns a namedtuple (values, indices)
        if min_sat < masks.max(0)[0].sum().float() / (h*w):
            # pick random masks
            start = np.random.randint(len(masks)) + 1
            for mask_num in range(start, len(masks)+1):
                sub_masks = masks[np.random.choice(len(masks), mask_num)]
                if min_sat < sub_masks.max(0)[0].sum().float() / (h*w):
                    break

        logging.debug('min_sat: {}, sub_masks: {}'.format(min_sat, sub_masks.max(0)[0].sum().float() / (h*w)))

        result_masks.append((sub_masks.max(0)[0].byte()* 255).unsqueeze(0))
        new_names.append(name)
    result_masks = torch.cat(result_masks)

    return result_masks, new_names

def load_flist(flist):
    if os.path.isfile(flist):
        try:
            return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        except:
            return [flist]
def save_masks(output, flist, masks):
    for file, mask in zip(flist,masks):
        logging.debug('{} save'.format(os.path.join(output, os.path.split(file)[1])))
        Image.fromarray(mask).save(os.path.join(output, os.path.split(file)[1]))

def preprocess(device,batch):
    images = []
    names = []
    for img in batch:
        try:
            image = Image.open(img)
        except:
            logging.debug('cant open {}'.format(img))
            continue
        images.append(np.transpose(np.array(image),(2,0,1)))
        names.append(img)
    images = np.array(images)
    result = torch.from_numpy(images).to(device=device).float() / 255
    print(result.size())
    return result,names
def mask(input = None, output = None):
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

    files = load_flist(args.input)
    for f in range(0,len(files),args.size):
        batch = files[f:f+args.size]
        # open이 안됐을 경우에 mask와 mapping 어그러짐
        images,names = preprocess(device,batch)
        assert len(images) == len(names)
        # mask를 dectection못했을 경우에 sum reduction 불가
        masks,names = masking(device,images,names)
        assert len(masks) == len(names)

        save_masks(args.output, names, masks.numpy())


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
    mask('test.flist','mask')