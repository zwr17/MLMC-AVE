import h5py

f = h5py.File("label2.h5", "r")
for key in f.keys():
    print(f[key].name)  
    print(f[key][:])
