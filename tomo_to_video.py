import sys
import mrcfile
import argparse
import cv2
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('input',type=str,help='Input tomograms in mrc format')
parser.add_argument('output',type=str,help='Output ptah of the movue')
parser.add_argument('--start',type=int,help='Starting frame of the tomogram')
parser.add_argument('--end',type=int,help='Ending frame of the tomogram')
parser.add_argument('--frame',type=int,help='Frame rate of the movie')
args = parser.parse_args()
mrc = mrcfile.open(args.input).data
print(mrc[0].shape)
print(mrc.shape[1],mrc.shape[2])


def write_movie(path,start,end,frame):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(path, fourcc, frame, (mrc.shape[1], mrc.shape[2]),isColor=False)
    for index in range(start,end):
        mean = mrc[index].mean()
        std = mrc[index].std()
        data = np.clip(mrc[index], mean - 2.5 * std, mean + 2.5 * std)
        data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        video.write(data)
    for index in range(end-1,start,-1):
        mean = mrc[index].mean()
        std = mrc[index].std()
        data = np.clip(mrc[index], mean - 2.5 * std, mean + 2.5 * std)
        data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        video.write(data)
    video.release()

    

if args.start == None:
    start = 0
else:
    start = args.start
    
if args.end == None:
    end = len(mrc)
else:
    end = args.end
write_movie(args.output,start,end,args.frame)
