import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from pycocotools.coco import COCO
import os
pylab.rcParams['figure.figsize'] = (8.0, 10.0) # image pixel

dataDir = r'/home/maxime/Documents/mathislab/bearproject/data/coco'
dataType = r'val'
annFile = r'{}/annotations/pred_keypoints_{}.json'.format(dataDir, dataType)

coco = COCO(annotation_file=annFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
nms = set([cat['supercategory'] for cat in cats])

catIds = coco.getCatIds(catNms=['dog'])
imgIds = coco.getImgIds(catIds=catIds)
index = np.random.randint(0, len(imgIds))
img = coco.loadImgs(imgIds[index])[0]  # Load images with specified ids.

image_path = os.path.join("/home/maxime/Documents/mathislab/bearproject/data/coco/images/val", img["file_name"])
i = io.imread(image_path)
plt.axis('off')
plt.imshow(i)
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
for ann in anns:
    kp = np.array(ann['keypoints'])
    print(kp)
    x = kp[0::3]
    y = kp[1::3]
    v = kp[2::3]
    plt.plot(x[v>0], y[v>0],'o',markersize=10, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
#coco.showAnns(anns)
plt.show()
