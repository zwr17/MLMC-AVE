# MLMC-AVE dataset

Wenru Zheng, Ryota Yoshihashi, Rei Kawakami, Ikuro Sato, and Asako Kanezaki.
**Multi Event Localization by Audio-Visual Fusion with Omnidirectional Camera and Microphone Array.**
*6th CVPR Workshop on Multimodal Learning and Applications (MULA)*, accepted, 2023.

MLMC-AVE is short for multi-label multi-channel audio-visual event.

This program is for the audio-visual event localization task for more real life in a school or an office. The dataset is made up of omnidirectional video and corresponding 8-channel audio. We film the omnidirectional video by theta-v and 8-channel audio by the Tamago device.
<div align=center><img width="550" src="https://github.com/zwr17/Multi-Event-Localization-by-Audio-Visual-Fusion-with-Omnidirectional-Camera-and-Microphone-Array/blob/main/device_dataset.png"/></div>

## Details
We defined 12 event categories assuming an indoor office environment with multiple sound sources. The defined categories are
man speaking, woman speaking, walking, typing, kettle boiling, writing on board, alarming, opening the door, opening the drawer, coughing, printer working,and cleaner working. There is also a category nothing for scenes without any event; thus, the total number of categories in our dataset is 13. 660 videos were collected, and the distribution of each category is shown in the following image. 
<div align=center><img width="500" src="https://github.com/zwr17/Multi-Event-Localization-by-Audio-Visual-Fusion-with-Omnidirectional-Camera-and-Microphone-Array/blob/main/category_distri.png"/></div>

And each video contains at least one AVE. Some examples of our dataset are shown here.
<div align=center><img width="650" src="https://github.com/zwr17/Multi-Event-Localization-by-Audio-Visual-Fusion-with-Omnidirectional-Camera-and-Microphone-Array/blob/main/example.png"/></div>

Dataset can be downloaded from https://drive.google.com/file/d/1jzmkA421L2zzs_VfBsCwELm7lq3Dt0Q-/view?usp=share_link . 
Features can be downloaded from https://drive.google.com/file/d/1k2jqt7pcXfP84iZbJpG5Jr5iqMvAVBsJ/view?usp=share_link .

The visual and audio inputs are individually pre-processed by pre-trained convolutional neural networks. To process the visual data, each frame image is extracted from the video and passed through ResNet pre-trained on ImageNet to obtain the frame-wise feature. For audio part, we first take the wav-format audio data from the video or the microphone(s) and compute the audio feature through the VGG-like CNN model pre-trained on the AudioSet dataset.

## Notice
We want to inform you that this dataset contains a person's face, so if you want to use or edit it, it is better to notify us of your aim.

