import os

#cut the audio into 15 seconds and change to signed-integer way
#also generate corresponding mono-channel audio


for i in range(487,590):

    a= str(i)

    os.system("sox "+a+ ".wav audio/"+a+".wav trim 00:00 00:15")
    os.system("sox audio/"+a+".wav -b 16 -e signed-integer audio16/"+a+".wav")
    os.system("sox audio16/"+a+".wav -c 1 audiomono/"+a+".wav")