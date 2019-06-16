import cv2
import numpy
import face_alignment
import numpy as np
import skimage.io as io
import os

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
predefined_color = [128,64,160,244,35,220,80,200]

dataset = '../../dataset/voxcelb1/data'
person_name = os.listdir(dataset)
num_classes = len(person_name)
video_path = [os.path.abspath(os.path.join(dataset,i)) for i in person_name]

# save all video clips
all_video_path = []
f = open('video_clips_path.txt','w')
print('saving video path!!')
for v in video_path:
    each_person_clip = os.path.join(v,'1.6')
    each_person_clip = [ os.path.join(each_person_clip,i,'1') for i in os.listdir(each_person_clip)]
    all_video_path += each_person_clip
    for path in each_person_clip:
        print(path)
        f.write(path+'\n')
f.close()
if not os.path.exists('../../dataset/voxcelb1/landmarks'):
    os.makedirs('../../dataset/voxcelb1/landmarks')

print('saving landmarks!!!')
f = open('video_landmarks_path.txt','w')
for v in all_video_path:
    print(v)
    for i in os.listdir(v):
        img = os.path.join(v,i)
        basename = os.path.dirname(img)
        filename = os.path.basename(img)
        input = io.imread(img)
        preds = fa.get_landmarks(input)
        if preds is None:
            continue
        preds = preds[-1]
        saved = np.ones_like(input,dtype=np.uint8)*255
        for i in range(17-1):
            cv2.line(saved,(preds[i,0],preds[i,1]),(preds[i+1,0],preds[i+1,1]),[predefined_color[0],250,10],2)
        for i in range(17,22-1):
            cv2.line(saved,(preds[i,0],preds[i,1]),(preds[i+1,0],preds[i+1,1]),[predefined_color[1],10,250],2)
        for i in range(22,27-1):
            cv2.line(saved,(preds[i,0],preds[i,1]),(preds[i+1,0],preds[i+1,1]),[predefined_color[2],150,150],2)
        for i in range(27,31-1):
            cv2.line(saved, (preds[i, 0], preds[i, 1]), (preds[i + 1, 0], preds[i + 1, 1]), [predefined_color[3],0,0],
                     2)
        for i in range(31,36-1):
            cv2.line(saved,(preds[i,0],preds[i,1]),(preds[i+1,0],preds[i+1,1]),[predefined_color[4],50,0],2)
        for i in range(36,42-1):
            cv2.line(saved,(preds[i,0],preds[i,1]),(preds[i+1,0],preds[i+1,1]),[predefined_color[5],0,50],2)
        for i in range(42,48-1):
            cv2.line(saved,(preds[i,0],preds[i,1]),(preds[i+1,0],preds[i+1,1]),[predefined_color[6],180,30],2)
        for i in range(48,60-1):
            cv2.line(saved, (preds[i, 0], preds[i, 1]), (preds[i + 1, 0], preds[i + 1, 1]), [predefined_color[7],20,180],
                     2)
        basename = basename.split('data')
        basename = basename[0] + 'dataset/voxcelb1/' 'landmarks' + basename[2]
        if not os.path.exists(basename):
            os.makedirs(basename)
        filename = os.path.join(basename,filename)
        print(filename)
        f.write(filename + '\n')
        cv2.imwrite(filename,saved)
f.close()


