import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import CANNet
import torch
from torch.autograd import Variable

from sklearn.metrics import mean_squared_error,mean_absolute_error

from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# the folder contains all the test images
img_folder='/home/toshiba_pc/Masaüstü/venice/test_data/images'     #directory will be venice dataset directory 
img_paths=[]

for img_path in glob.glob(os.path.join(img_folder, '*.jpg')):
    img_paths.append(img_path)

model = CANNet()

model = model.to(device)

checkpoint = torch.load('checkpoint.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred= []
gt = []

for i in range(len(img_paths)):
    img = transform(Image.open(img_paths[i]).convert('RGB')).to(device)
    img = img.unsqueeze(0)
    h,w = img.shape[2:4]
    h_d = int(h/2)
    w_d = int(w/2)
    img_1 = Variable(img[:,:,:h_d,:w_d].to(device))
    img_2 = Variable(img[:,:,:h_d,w_d:].to(device))
    img_3 = Variable(img[:,:,h_d:,:w_d].to(device))
    img_4 = Variable(img[:,:,h_d:,w_d:].to(device))
    density_1 = model(img_1).data.cpu().numpy()
    density_2 = model(img_2).data.cpu().numpy()
    density_3 = model(img_3).data.cpu().numpy()
    density_4 = model(img_4).data.cpu().numpy()

    pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground-truth'),'r')   #original is ground_truth
    groundtruth = np.asarray(gt_file['density'])
    pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()
    pred.append(pred_sum)
    number_gt=np.sum(groundtruth)
    gt.append(number_gt)

    print("For image",i,"predicted number is",pred_sum,"real number is",number_gt)
    
		
mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

print ('MAE: ',mae)
print ('RMSE: ',rmse)
