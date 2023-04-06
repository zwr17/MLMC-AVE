import os

#for separating the long video into 15-seconds ones


for i in range(464,500):
    b=i+90
    a= str(i)
    c=str(b)
    os.system("ffmpeg -i video/R0010"+ a +".mp4 -ss 0 -t 15 -codec copy video15/"+ c +".mp4")