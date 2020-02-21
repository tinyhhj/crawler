import numpy as np
import torchvision.models as models
import torch
import os

debug = True
orig_print = print
def print(*args, **kwargs):
    orig_print(*args, **kwargs)

def masking(images,th= 0.3):
    """
    random object masking from image
    :param image: tensor image (c,h,w)
    :return: randomly object masked image (h,w)
    """
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    batch_size = images.size()[0]
    h = images.size()[2]
    w = images.size()[3]
    result_masks = torch.zeros((batch_size,h,w),dtype=torch.uint8,device=device)

    model  = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)

    model.eval()
    results = model(images.to(device))

    for i,result in enumerate(results):
        print(result['masks'].size())
        masks = result['masks'].squeeze() >= th
        print(result_masks[i,:,:].type())
        for m in masks[torch.arange(np.random.randint(len(masks)))]:
            result_masks[i,:,:] = torch.where(m, torch.tensor(255,device=device,dtype=torch.uint8), result_masks[i,:,:])
    return result_masks.to('cpu')

def load_flist(flist):
    if os.path.isfile(flist):
        try:
            return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        except:
            return [flist]

#
if __name__ =='__main__':
    flist = load_flist('test.flist')
    from PIL import Image
    t = torch.empty(0)
    for f in flist:
        t= torch.cat((t,torch.from_numpy(np.transpose(np.array(Image.open(f)),(2,0,1))).to(dtype=torch.float).unsqueeze(0)))
    print(t.size())
    r = masking(t/255).numpy()
    import matplotlib.pyplot as plt
    os.makedirs('mask',exist_ok=True)
    for i,m in enumerate(r):
        Image.fromarray(m).save('mask/{}'.format(os.path.split(flist[i])[1]))