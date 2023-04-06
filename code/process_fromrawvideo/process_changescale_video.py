import os

#change the scale of each video into 640X320


for i in range(1,669):
    #b=i+90
    a= str(i)
    #c=str(b)
    os.system("ffmpeg -i video15/"+a+".mp4 -vf scale=640:320 video640320/"+a+".mp4 -hide_banner")