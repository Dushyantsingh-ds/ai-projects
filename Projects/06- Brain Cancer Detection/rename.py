import os
os.chdir('dat')
i=584
for src in os.listdir():
    dst="BrainCancer"+"_"+str(i)+".jpg"
    os.rename(src,dst)
    i+=1

