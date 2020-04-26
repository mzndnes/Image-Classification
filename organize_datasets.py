import os
import shutil

tr_t=1
if tr_t==0:
    dirname='train'
else:
    dirname='test'

img_dir="cropped"
rut="act_"
curpath = os.path.abspath(os.getcwd())


tr_t=os.path.join(curpath,'images',rut+dirname)


try:
    os.mkdir(tr_t)
except:
    print('Exists')

src_dir=os.path.join(curpath,'images',img_dir)


dirs=os.listdir(src_dir)

for dr in dirs:
    sub_dir = os.path.join(src_dir, dr)
    dst_dir=os.path.join(tr_t, dr)

    try:
        os.mkdir(dst_dir)
    except:
        print('Exists')
    cnt=0
    files=os.listdir(sub_dir)
    for fl in files:
        if cnt==5:
            break
        dfl = os.path.join( dst_dir,fl)
        sfl = os.path.join(sub_dir, fl)

        shutil.move(sfl, dfl)
        cnt+=1





