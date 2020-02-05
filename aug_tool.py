# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=C0326
# pylint: disable=C0330
# pylint: disable=C0305
# pylint: disable=C0301
# pylint: disable=C0303
import cv2
import math
import random
import warnings
import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage import exposure
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
from sklearn.datasets import load_files
from sklearn.preprocessing import LabelBinarizer


class data_aug():


    def __init__(self,image_files=None,annotations=None,targets=None,batch_size=32,
                 shuffle=True,rot_range=0,
                 h_flip=False,v_flip=False,
                 horizontal_shift_range=0,
                 vertical_shift_range=0,
                 horizontal_shear=0,vertical_shear=0,
                 scale=0,luminance=0,clip_limit=20,
                 adaptive_histogram_equalisation=False,
                 generate_images=False,
                 copies_per_image=1,
                 contrast=0):

        self.image_files = image_files
        self.annotations = annotations
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rot_range = rot_range
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.curr_ind = 0
        self.horizontal_shift_range = horizontal_shift_range
        self.vertical_shift_range = vertical_shift_range
        self.horizontal_shear = horizontal_shear
        self.vertical_shear = vertical_shear
        self.scale = scale
        self.luminance = luminance
        self.clip_limit = clip_limit
        self.adap_histeq = adaptive_histogram_equalisation
        self.contrast = contrast
        self.generate_images = generate_images
        self.copies_per_image = copies_per_image
        iter(self)

    def rotation(self,theta):
        matrix = np.array([[np.cos(theta),-np.sin(theta),0],
                       [np.sin(theta),np.cos(theta),0],
                       [0,0,1]]) #rotational matrix
        return matrix

    def hflip(self,im):
        x = random.random()
        #print(x)
        img = im
        if x>=0.67: #for random flipping if the value is greater than 0.67 then only the image is flipped
            img = np.fliplr(im)
        return  img #horizontal flip

    def vflip(self,im):
        y = random.random()
        img = im
        if y>=0.67 :   #for random flipping if the value is greater than 0.67 then only the image is flipped
            img = np.flipud(img)
        return  img #vertical flip

    def im_shift(self,tx,ty):
        matrix = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
        return matrix #translational matrix

    def shear_y(self,val):
        matrix = np.array([[1,np.sin(val),0],[0,1,0],[0,0,1]])
        return matrix #horizontal shear matrix

    def shear_x(self,val):
        matrix = np.array([[1,0,0],[np.sin(val),1,0],[0,0,1]])
        return matrix #vertical shear matrix

    def im_scale(self,sx,sy):
        matrix = np.array([[sx,0,0],[0,sy,0],[0,0,1]]) # for value greater than 1 zoom out , less than 1 zoom in
        return matrix  #scaling matrix

    def brightness(self,im,gam):
        #print(type(im))
        im = im.astype('float32')
        new_im = exposure.adjust_gamma(im,gam)
        return new_im

    def ada_hist_eq(self,im,cl): # adaptive histogram equalisation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_im = exposure.equalize_adapthist(im,clip_limit=cl)

        return new_im

    def contra(self,im,con):
        im = im.astype('float32')
        new_im = exposure.adjust_sigmoid(im,con)  # con vakue between 0 and 1
        return new_im

    def offset_center(self,mat,ht,wd):
        x = float(wd)/2
        y = float(ht)/2
        offset = np.array([[1,0,y],[0,1,x],[0,0,1]]) #translation matrix
        reset = np.array([[1,0,-y],[0,1,-x],[0,0,1]]) #translation matrix
        result = np.dot(np.dot(offset,mat),reset)
        return result

    def apply_transform(self,image,matrix,channel_index=2):
        x = np.rollaxis(image,channel_index,0)
        fm = matrix[:2,:2]
        fo = matrix[:2,2]
        ci = [ndi.interpolation.affine_transform(xc,fm,fo,order=0,mode='nearest',cval=255)
            for xc in x]
        x = np.stack(ci,axis=0)
        x = np.rollaxis(x,0,channel_index+1)
        return x

    def get_transformed_image(self,image):
        h = image[0].shape[0]
        w = image[0].shape[1]
        inp_image = image[0]
        op_image = image[1] if self.annotations is not None else None
        mod_inp = image[0].copy()
        mod_op = image[1].copy() if self.annotations is not None else None
        tf_mat = None

        if self.rot_range:
            rot = np.pi/180 * np.random.uniform(-self.rot_range,self.rot_range)
        else:
            rot = 0

        if self.horizontal_shift_range:
            ty = np.random.uniform(-self.horizontal_shift_range,self.horizontal_shift_range)

        else:
            ty = 0

        if self.vertical_shift_range:
            tx = np.random.uniform(-self.vertical_shift_range,self.vertical_shift_range)
        else:
            tx = 0

        if self.horizontal_shear:
            x_shear = np.pi/180 *np.random.uniform(-self.horizontal_shear,self.horizontal_shear)

        else:
            x_shear = 0

        if self.vertical_shear:
            y_shear = np.pi/180 *np.random.uniform(-self.vertical_shear,self.vertical_shear)
        else:
            y_shear = 0

        if self.scale:
            sca = np.random.uniform(0,self.scale)

        else:
            sca = 0

        if rot!=0:
            rot_mat = self.rotation(rot)
            tf_mat = rot_mat

        if tx!=0 or ty!=0:
            shift_mat = self.im_shift(tx,ty)
            tf_mat = shift_mat if tf_mat is None else np.dot(tf_mat,shift_mat)

        if x_shear!=0:
            shx_mat = self.shear_x(x_shear)
            tf_mat = shx_mat if tf_mat is None else np.dot(tf_mat,shx_mat)

        if y_shear!=0:
            shy_mat = self.shear_y(y_shear)
            tf_mat = shy_mat if tf_mat is None else np.dot(tf_mat,shy_mat)

        if sca!=0:
            sca_mat = self.im_scale(sca,sca)
            tf_mat = sca_mat if tf_mat is None else np.dot(tf_mat,sca_mat)

        if tf_mat is not None:
            off_mat = self.offset_center(tf_mat,h,w)

            mod_inp = self.apply_transform(inp_image,off_mat)
            #print(mod_inp)

            mod_op = self.apply_transform(op_image,off_mat) if self.annotations is not None else None



        if self.h_flip:
            mod_inp = self.hflip(mod_inp)
            mod_op = self.hflip(mod_op) if self.annotations is not  None else None

        if self.v_flip:
            mod_inp = self.vflip(mod_inp)
            mod_op = self.vflip(mod_op) if self.annotations is not None else None

        if self.luminance:
            gamma = np.random.uniform(0,self.luminance)
            #print(mod_inp)
            mod_inp = self.brightness(mod_inp,gamma)
            mod_op = self.brightness(mod_op,gamma) if self.annotations is not None else None

        if self.adap_histeq:
            clip = np.random.uniform(0,self.clip_limit)
            mod_inp = self.ada_hist_eq(mod_inp,clip)
            mod_op = self.ada_hist_eq(mod_op,clip) if self.annotations is not None else None

        if self.contrast:
            cont = np.random.uniform(0,self.contrast)
            mod_inp = self.contra(mod_inp,cont)
            mod_op = self.contra(mod_op,cont) if self.annotations is not None else None

        return mod_inp,mod_op

    def get_batches(self,imgs,bz,classes):
        cpi = self.copies_per_image
        if self.shuffle == True:
            imgs = random.sample(imgs,bz) #shuffles the images
            imgs = np.asarray(imgs)
            tar = imgs[:,2] if self.targets is not None else None
            imgs = imgs[:,:2]
        else:
            imgs = np.asarray(imgs)
            tar = imgs[:,2] if self.targets is not None else None
            imgs = imgs[:,:2]

        batch_img = []
        if self.generate_images == True:
            med = np.zeros(shape=(bz*cpi,classes[0]),dtype=np.float32) if self.targets is not None else None
            x = 0
            for i,j in enumerate(imgs):
                for ab in range(cpi):
                    batch_img.append(self.get_transformed_image(j)) #transforms both input and annotated image
                    if self.targets is not None:
                        med[x] = tar[i]
                    x = x + 1


        else:
            med = np.zeros(shape=(bz,classes[0]),dtype=np.float32) if self.targets is not None else None
            for i,j in enumerate(imgs):
                batch_img.append(self.get_transformed_image(j))
                if self.targets is not None:
                    med[i] = tar[i]


        batch_in,batch_out = zip(*batch_img)
        out = {}
        out['input'] = np.asarray(batch_in,dtype='float32')
        out['output'] = np.asarray(batch_out,dtype='float32')
        out['target'] = med
        return out






    def __iter__(self):
        return self

    def __next__(self):
        siz = len(self.image_files)
        sind = self.curr_ind
        if siz - sind >= self.batch_size:
            bs = self.batch_size
        else:
            bs = siz - sind
        lind = sind + bs
        in_list = []
        for i in range(sind,lind):
            im = self.image_files[i]
            op = self.annotations[i] if self.annotations is not None else None
            tg = self.targets[i] if self.targets is not None else None
            emb = (im,op,tg)
            in_list.append(emb)

        sha = self.targets[0].shape if self.targets is not None else None
        batch = self.get_batches(in_list,bs,sha)
        self.curr_ind = lind
        return batch

    def next(self):
        bat = self.__next__()
        return bat






def load_data(fold_path):
    data = load_files(fold_path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    one_hot = LabelBinarizer()
    ohd = one_hot.fit_transform(targets)
    return files, ohd

def extract(img,rsz=None):
    print(img)
    imag = plt.imread(img)
    #imag = cv2.resize(imag,(224,224))
    channels = imag.shape[2]
    if channels == 4:
        imag = cv2.cvtColor(imag,cv2.COLOR_RGBA2RGB)
    elif channels == 1:
        imag = cv2.cvtColor(imag,cv2.COLOR_GRAY2RGB)
    #elif channels == 3:
        #imag = cv2.cvtColor(imag,cv2.COLOR_BGR2RGB)


    if rsz is not None:
        imag = cv2.resize(imag,rsz)
    if len(imag.shape) == 2:
        imag = np.expand_dims(imag,axis=2)   ##GrayScale
    return imag

def tensor_4d(img_fol,rsh):
    list_of_tensors = [extract(im,rsh) for im in tqdm(img_fol)]
    return np.stack(list_of_tensors,axis=0)

def one_hot_encode(targets):
    one_hot = LabelBinarizer()
    ohd = one_hot.fit_transform(targets)
    return ohd
