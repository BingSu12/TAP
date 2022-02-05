import os
import glob
import shutil

import numpy as np

max_seq_len = 8

old_dir = "data/ssv2_256x256q5/val"
new_dir = "data/ssv2_l{}/val".format(max_seq_len)

# old_dir = "data/ucf_256x256q5/val"
# new_dir = "data/ucf101_l{}/val".format(max_seq_len)

# old_dir = "data/hmdb_256x256q5/train"
# new_dir = "data/hmdb_l{}/train".format(max_seq_len)

classes = [j for j in glob.glob(os.path.join(old_dir, "*"))]

print(classes)

os.mkdir(new_dir)

src_jpgs = 0
tgt_jpgs = 0

for class_folder in classes:
    c = os.path.split(class_folder)[-1]
    new_c = os.path.join(new_dir, c)
    print(new_c)
    os.mkdir(new_c)
    for video_folder in glob.glob(os.path.join(class_folder, "*")):
        v  = os.path.split(video_folder)[-1]
        print(v)
        new_v = os.path.join(new_c, v)

        jpgs = [j for j in glob.glob(os.path.join(video_folder, "*.jpg"))]

        n_jpgs = len(jpgs)
        print(n_jpgs)
        src_jpgs += n_jpgs

        if n_jpgs <= max_seq_len:
            tgt_jpgs += n_jpgs
            shutil.copytree(video_folder, new_v)
        else:
            os.mkdir(new_v)
            jpgs.sort()

            idx_f = np.linspace(0, n_jpgs-1, num=max_seq_len)
            idxs = [int(f) for f in idx_f]

            for i in range(len(idxs)):
                src = jpgs[idxs[i]]
                tgt = os.path.join(new_v, "{:08d}.jpg".format(i+1))
                shutil.copy(src, tgt)
                tgt_jpgs += 1

print("Reduced {} to {} jpgs".format(src_jpgs, tgt_jpgs))