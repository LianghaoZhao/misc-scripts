import mrcfile
import sys
import numpy as np
import os.path
import shutil
path = sys.argv[1]
new_path = os.path.split(path)[1].replace('full_rec.mrc','') + 'rec_for_isonet.mrc'
slice_range = (int(sys.argv[2])-1,int(sys.argv[3])-1)

shutil.copyfile(path,new_path)

mrc = mrcfile.open(new_path,mode='r+')
data = np.swapaxes(mrc.data,0,1)
data = data[slice_range[0]:slice_range[1],:,:]
mrc.set_data(data)
mrc.close()

