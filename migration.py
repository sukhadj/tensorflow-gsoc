import os

IN = "/home/sukhad/Workspace/GSoC19/Tensorflow/models/official/utils/testing/"
OUT = "/home/sukhad/Workspace/GSoC19/Tensorflow/Tensorflow-models-2.0/ResNet/utils/testing/"

files = os.listdir(IN)

#print(files)

for file in files:
    if ".py" in file:
        os.system("tf_upgrade_v2 --infile "+IN+file+" --outfile "+OUT+file)
        print(file)
