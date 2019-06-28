import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy import interpolate
import pickle
import glob
import os


def create_dummy_points(alt,pitch):

    #create an array of the distances given the camera alt and pitch
    points=np.zeros((960,1280,3))
    x=points[:,:,1]
    y=points[:,:,0]
    z=points[:,:,2]
    
    #find the extrmea
    x_2=alt*(math.sin(math.radians(90-10-pitch)))/math.sin(math.radians(10+pitch))
    x_1=alt*(math.sin(math.radians(90-40-pitch)))/math.sin(math.radians(40+pitch))
    start=x_2

    #find the y step size
    step_y=(x_2)/960

    #loop through the array
    for i in range(960):
        x[i]=start
        start-=step_y
        

    start=-1*(x_2*math.tan(math.radians(40)))
    step_x=(start*-2)/1280    

    for i in range(1280):
        y[:, i]=start
        start+=step_x

    points_test=np.dstack((x,y))
    points_test=np.dstack((points_test,z))
    
    return points_test


def segment_image(img):

    #segment the image based on black
    
    #threshhold image to find object
    th, im_th = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV);
 
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    
    return im_out
    

def bounding_box(segment):

    #get a bounding box of the segmented object
    
    #extract non-zero points from segment i.e. hot points
    nonzero = segment.nonzero()
    yp = np.array(nonzero[0])
    xp = np.array(nonzero[1])

    #combine non zero points
    X=np.column_stack((xp,yp))

    #use min area rect
    rect=cv2.minAreaRect(X)
    box=cv2.boxPoints(rect)
    box = np.int0(box)
    
    return box


def get_rotation(segment):

    #use PCA to find the angle of the object
    
    #extract non zero points
    nonzero = segment.nonzero()
    yp = np.array(nonzero[0])
    xp = np.array(nonzero[1])

    #combine the points for PCA
    X=np.column_stack((xp,yp))

    #perfornm PCA
    pca = PCA(n_components=2)
    pca.fit(X)

    # plot data
    points=[]
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        #points.append((pca.mean_, pca.mean_ + v))
        points.append(pca.mean_ + v)


    #centroid
    x=[pca.mean_[0],points[0][0]]
    y=[pca.mean_[1],points[0][1]]
    
    #extrema
    delta_x=points[0][0]-pca.mean_[0]
    delta_y=points[0][1]-pca.mean_[1]
    
    rotation=math.atan(delta_x/delta_y)
    rotation=(90-math.degrees(rotation))*1
    
    #convert to axis
    if rotation > 90:
        rotation=180-rotation
        
    else:
        rotation=rotation*-1
    
    return rotation
    
    
def get_range(box,angle):
    
    #object constant
    short_side=.5
    long_side=.2
    
    #create a range reference array and parse it out
    points=create_dummy_points(1,0)
    y_real=points[:,:,0]
    x_real=points[:,:,1]
    
    #find the middle of the bottom bounding box line
    delta_x=box[1][0]-box[0][0]
    delta_y=box[1][1]-box[0][1]
    
    #get the length of the bottom line
    len_1=math.sqrt(delta_x**2 + delta_y**2)
    
    #find edge location
    x_cg=int(box[0][0]+0.5*delta_x)
    y_cg=int(box[0][1]+0.5*delta_y)
    
    #get the dimensions of the side line
    delta_x_2=box[1][0]-box[2][0]
    delta_y_2=box[1][1]-box[2][1]
    
    #get the length of the side line
    len_2=math.sqrt(delta_x_2**2 + delta_y_2**2)
    
    
    #if len_1 is bigger then the bottom line is the long side
    if len_1 >= len_2:    #use long side
        
        #check angle
        if angle <= 0:
            x,y = get_deltas(90+angle,long_side)
            y=abs(y)
            
        elif angle > 0:
            x,y = get_deltas(angle,long_side)
            
        print x,y,"long"
    
    #else len_1 is the short side
    else:   #use short side
        
        #check angle
        if angle <= 0:
            x,y = get_deltas(90+angle,short_side)
            y=abs(y)
            
        elif angle > 0:
            x,y = get_deltas(angle,short_side)
            
        print x,y,"short"
        
    
    real_loc_x=x_real[y_cg][x_cg]+x
    real_loc_y=y_real[y_cg][x_cg]+y
        
    return real_loc_x,real_loc_y

def show_box(img,box):
    #plot the bounding box


    cv2.circle(img,(box[0][0], box[0][1]), 35, (255,0,0), -1)
    #cv2.circle(img,(x, y), 35, (255,0,0), -1)
    cv2.drawContours(img, [box], 0, (0,0,255), 2)
    plt.imshow(img)
    plt.show()
    
def get_deltas(angle,object_const):

    #find the real distance to the centroid of the object from the edge 
    
    #convert to radians
    angle=math.radians(angle)
    
    delta_x=object_const*math.cos(angle)
    delta_y=object_const*math.sin(angle)
    
    return delta_x,delta_y




#read in image
img=cv2.imread("/home/avengers/Desktop/test_3.jpg",0)
img=cv2.resize(img,(1280,960))

#segment the image
seg=segment_image(img)

#get segmentation shape
rows,cols = seg.shape

#rotate the segmentation
M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
dst = cv2.warpAffine(seg,M,(cols,rows))

#show the rotated segmentation
plt.imshow(dst)
plt.show()

#get bounding box
box=bounding_box(dst)

#get the angle of the object
angle=get_rotation(dst)

#get the distance in cartisian from the camera
range_x,range_y=get_range(bounding_box(dst),angle)

#plot the box
show_box(img,box)

