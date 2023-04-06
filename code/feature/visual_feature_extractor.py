import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
import numpy as np
import h5py
import sys
import cv2
import pylab
import imageio
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

pretrained = True
original_resnet = torchvision.models.resnet18(pretrained)


def video_frame_sample(frame_interval, video_length, sample_num):
    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    return num

class VisualNet(nn.Module):
    def __init__(self, original_resnet):
        super(VisualNet, self).__init__()
        #modules = list(resnet.children())[:-1]
        layers = list(original_resnet.children())[:-2]
        self.feature_extraction = nn.Sequential(*layers) #features before conv1x1

    def forward(self, x):
        x = self.feature_extraction(x)
        return x

pretrained = True
original_resnet = torchvision.models.resnet18(pretrained)
model = VisualNet(original_resnet)

#base_model = VGG19(weights='imagenet')
#base_model = (weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output) # vgg pool5 features

# path of your dataset
#video_dir = "./15s video/"
video_dir = "./datasetupdate/video640320"
lis = os.listdir(video_dir)

lis.sort(key=lambda x:int(x[:-4]))

#lis = lis.sort(key=lambda x: int(x.split('.')[0]))

len_data = len(lis)
#video_features = np.zeros([len_data, 10, 7, 7, 512]) # 10s long video
video_features = np.zeros([len_data, 15, 8, 16, 512])
t = 15 # length of video
sample_num = 16 # frame number for each second
#print(len_data)--->181

c = 0
for num in range(len_data):

    '''feature learning by VGG-net'''
    #video_index = os.path.join(video_dir, lis[num] + '.mp4') # path of videos
    video_index = os.path.join(video_dir, lis[num])  # path of videos
    print(video_index)

    vid = imageio.get_reader(video_index, 'ffmpeg')
    #vid_len = len(vid)
    vid_len = vid.count_frames()
    #print(vid_len) 450
    #frame_num = vid.count_frames()
    #print(vid_len)
    frame_interval = int(vid_len / t) #30
    #print(frame_interval)

    frame_num = video_frame_sample(frame_interval, t, sample_num)
    #print(frame_num)
    imgs = []
    for i, im in enumerate(vid):
        x_im = cv2.resize(im, (512, 256))  #512 256

        imgs.append(x_im)
    vid.close()
    extract_frame = []
    for n in frame_num:
        extract_frame.append(imgs[n])

    #feature = np.zeros(([15, 16, 7, 14, 512]))
    feature = np.zeros(([15, 16, 8, 16, 512])) #adapt for 8 channel audio
    #feature = np.zeros(([15, 16, 7, 7, 512]))
    print(len(extract_frame)) #
    for j in range(len(extract_frame)):
        y_im = extract_frame[j]

        y_im2 = torch.from_numpy(y_im.astype(np.float32)).clone()

        #y_im2 = image.img_to_array(y_im)
        #x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        #pool_features = np.float32(model.predict(x))
        #pool_features = torch.from_numpy(model(x))
        #a = model(x).cpu()

        x = Variable(torch.unsqueeze(y_im2, dim=0).float(), requires_grad=False)
        # y1 = resnet18(x)
        x = x.permute(0, 3, 1, 2)
        x = x.cuda()
        model = model.cuda()
        #print(x.size()) #[1, 3, 224, 448]
        y = model(x).cpu()
        y = y.data.numpy()
        #np.savetxt(feature_path, y, delimiter=',')
        #print(y)
        #print(y.shape) #(1, 512, 7, 14)
        y = np.transpose(y, (0,2,3,1))
        #y = y.permute(0, 1, 2, 3) #(1, 7, 14，512)

        tt = int(j / sample_num)
        video_id = j - tt * sample_num
        feature[tt, video_id, :, :, :] = y
    feature_vector = np.mean(feature, axis=(1)) # averaging features for 16 frames in each second
    video_features[num, :, :, :, :] = feature_vector
    c += 1
    print(c)

# save the visual features into one .h5 file. If you have a very large dataset, you may save each feature into one .npy file
with h5py.File('./video_cnn_feature_update.h5', 'w') as hf:
    hf.create_dataset("dataset", data=video_features)
