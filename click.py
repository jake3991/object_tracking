import numpy as np
# import the necessary packages
import argparse
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = int(255)
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #mask=cv2.bitwise_not(mask)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    masked_image=cv2.addWeighted(masked_image,1.0,cv2.bitwise_not(mask),1.0,0)
    
    return masked_image

def segment_image(img,thr):

    #segment the image based on black
    
    #threshhold image to find object
    th, im_th = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY_INV);
 
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

def segment_image_2(channel,thresh_min,thresh_max):

    #segment an image based on a top and bottom threshold

    #create an image of zeros
    binary = np.zeros_like(channel)

    #fill all pixels inside the threshold with white
    binary[(channel >= thresh_min) & (channel <= thresh_max)] = 255
    
    return binary

def morph_segment(img):

    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    return erosion

def remove_segment_noise(img,count):

    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    nonzero = erosion.nonzero()
    yp = np.array(nonzero[0])
    xp = np.array(nonzero[1])
    X=np.column_stack((xp,yp))

    db = DBSCAN(eps=3, min_samples=2).fit(X)
    labels = db.labels_

    dbscan_out=np.column_stack((labels,X))
    labels=list(labels)
    labels_unique=list(set(labels))
    labels_unique.sort()
    if labels_unique[0] == -1:
        labels_unique.pop(0)

    largest_cluster=labels_unique[0]

    for label in labels_unique:

        if labels.count(label) > labels.count(largest_cluster):
            largest_cluster = label

    #if len(labels_unique) == 2:
    #    for label in labels_unique:
    #        print label,labels.count(label)

    dbscan_out = dbscan_out[dbscan_out[:, 0] == 0]
    xp=dbscan_out[:, 1]
    yp=dbscan_out[:, 2]
    image = np.zeros((960, 1280))
    image[yp,xp] = 255

    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(image,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)


    return closing

def bounding_box(segment):

    #get a bounding box around a cluster of points

    #conver to array
    segment=np.array(segment)
    
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

def show_box(img,box):

    #plot the bounding box

    #draw the box
    cv2.drawContours(img, [box], 0, (0,0,255), 2)
    
    #show the image with the drawn box
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img
    
def plot_box(img,box):

    #plot the bounding box

    #draw the box
    cv2.drawContours(img, [box], 0, (0,0,255), 2)
    
    return img

def show_boxes(img,box_1,box_2):

    #show 2 boxes

    #plot the bounding boxes
    cv2.drawContours(img, [box_1], 0, (0,0,255), 2)
    cv2.drawContours(img, [box_2], 0, (0,0,255), 2)
    
    #show the image with boxes drawn
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def shift_axis(box,x,y):

    #shift axis from local box frame to global image frame

    #loop through 4 points in the box and shift them
    for i in range(0,len(box)):
         box[i][0] = x+box[i][0]
         box[i][1] = y+box[i][1]
         
    return box

def expand_long_side(box,size):

    #grow the box given that the longest side is on the bottom

    #find the current size of the bounding box
    delta_x=box[1][0]-box[0][0]
    delta_y=box[1][1]-box[2][1]

    
    #if delta x is negative flip the sign and move cooridnates
    if delta_x < 0:
    
        change_x=abs((delta_x*size - delta_x)/2)
        box[0][0]+=int(change_x)
        box[1][0]-=int(change_x)
        box[2][0]-=int(change_x)
        box[3][0]+=int(change_x)

    #if delta x is positive do not flip the sign and move cooridnates
    elif delta_x > 0:
        
        change_x=(delta_x*size - delta_x)/2
        box[0][0]-=int(change_x)
        box[1][0]+=int(change_x)
        box[2][0]==int(change_x)
        box[3][0]-=int(change_x)


    #adjust the y values
    change_y=(delta_y*size - delta_y)/2
    box[0][1]+=int(change_y)
    box[1][1]+=int(change_y)
    box[2][1]-=int(change_y)
    box[3][1]-=int(change_y)

    return box

def expand_short_side(box,size):

    #grow the box given that the shorest side is on the bottom

    #find the current size of the bounding box
    delta_x=box[3][0]-box[0][0]
    delta_y=box[0][1]-box[1][1]



    #if delta x is negative flip the sign and move cooridnates
    if delta_x < 0:
        
        change_x=abs((delta_x*size - delta_x)/2)
        box[0][0]+=int(change_x)
        box[1][0]-=int(change_x)
        box[2][0]-=int(change_x)
        box[3][0]+=int(change_x)

    #if delta x is positive do not flip the sign and move cooridnates
    elif delta_x > 0:
        
        change_x=(delta_x*size - delta_x)/2
        box[0][0]-=int(change_x)
        box[1][0]-=int(change_x)
        box[2][0]+=int(change_x)
        box[3][0]+=int(change_x)


    #adjust the y values
    change_y=(delta_y*size - delta_y)/2
    box[0][1]+=int(change_y)
    box[1][1]-=int(change_y)
    box[2][1]-=int(change_y)
    box[3][1]+=int(change_y)
    
    return box

def get_range(box,pitch,alt):

    #get the distance in meters to the object given the bounding box
    #note x is forward and y is postive right
    
    #find the middle of the bottom bounding box line
    delta_x=box[1][0]-box[0][0]
    delta_y=box[1][1]-box[0][1]
    
    #get the length of the bottom line
    len_1=math.sqrt(delta_x**2 + delta_y**2)
    
    #find edge location
    x_middle=int(box[0][0]+0.5*delta_x)
    y_middle=int(box[0][1]+0.5*delta_y)
    
    #get angle of pixel
    beta = (64/960.) * x_middle
    phi = ((80/1280.) * y_middle) - 40

    #get x distance
    x_real = alt * math.tan(math.radians(-9.5+pitch+beta))

    #get y_distance
    y_real = alt * (math.tan(math.radians(phi)))

    return x_real,y_real

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
        points.append(pca.mean_ + v)


    #centroid
    x=[pca.mean_[0],points[0][0]]
    y=[pca.mean_[1],points[0][1]]
    
    #extrema
    delta_x=points[0][0]-pca.mean_[0]
    delta_y=points[0][1]-pca.mean_[1]
    
    #get rotation
    rotation=math.atan(delta_x/delta_y)
    rotation=(90-math.degrees(rotation))*1
    
    #shift axis so zero rotation is x axis straight up the center of the image
    if rotation > 90:
        rotation=180-rotation
    else:
        rotation=rotation*-1
    
    return rotation,x,y

def expand_box(box,size):
    print "input",box
    #print "in",box

    #expand the box size, check which side (long or short) is on bottom
    #and call the appropriate function

    #find the middle of the bottom bounding box line
    delta_x=box[1][0]-box[0][0]
    delta_y=box[1][1]-box[0][1]
    
    #get the length of the bottom line
    len_1=math.sqrt(delta_x**2 + delta_y**2)
        
    #get the dimensions of the side line
    delta_x_2=box[1][0]-box[2][0]
    delta_y_2=box[1][1]-box[2][1]
    
    #get the length of the side line
    len_2=math.sqrt(delta_x_2**2 + delta_y_2**2)

    #if the long side is down call long side
    if len_1 > len_2:
        #print "long"
        box = expand_long_side(box,size)

    #otherwise call short side down
    else:
        box = expand_short_side(box,size)
        #print "short"

    #return the box in ROI format
    print "out",box
    return np.array([box], dtype=np.int32)

def init_box(points,filenames):

    #take initial selected bounding box and develop a true bounding box

    #read in the image as gray and convert it
    image = cv2.imread(filenames[0],0)
    image = cv2.resize(image,(1280,960))
    rows,cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
    image = cv2.warpAffine(image,M,(cols,rows))

    #crop out the image to show it
    crop_img = image[points[0][1]:points[1][1], points[0][0]:points[1][0]]

    #blur the image
    crop_img = cv2.GaussianBlur(crop_img, (5, 5), 0)

    #segment the image
    #seg = segment_image(crop_img,70)
    seg = segment_image_2(crop_img,80,115)

    #get the bounding box
    box = bounding_box(seg)

    #convert to BGR
    crop_img_color = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)

    #show the bounding box
    show_box(crop_img_color,box)

    #shift the coordinate axis of the bounding box to the global frame
    box =np.array([shift_axis(box,points[0][0],points[0][1])], dtype=np.int32)

    #white out areas ouside the reigon of interest
    display = region_of_interest(image,box)

    #in the next timestep get a 20% larger bouding box
    initial_guess_box = expand_box(box[0],1.2)

    #black out areas ouside the reigon of interest
    display=region_of_interest(image,initial_guess_box)

    return initial_guess_box

def search_frame(initial_guess_box,name,count,vis=True):

    #search for the new bouding box given an initial guess

    #read in the image
    image = read(name,0)

    if int(count) > 125:
        
        print "initial guess",count
        #show_box(image,initial_guess_box)

    #image speficic
    #image=region_of_interest(image,np.array([[[165,960],[165,0],[1120,0],[1120,960]]], dtype=np.int32))

    #search in the area near the old object
    initial_guess=region_of_interest(image,[initial_guess_box])

    #apply a guassian blur
    initial_guess_blur = cv2.GaussianBlur(initial_guess, (5, 5), 0)

    #segment the image
    #seg = segment_image(initial_guess_blur,70)
    seg = segment_image_2(initial_guess_blur,50,105)
    

    seg = remove_segment_noise(seg,count)

    cv2.imwrite("/home/avengers/Desktop/centroid_locator/seg/"+str(count)+".jpg",seg)
     


    #get ortientation
    theta, x, y = get_rotation(seg)

    #get the bounding box
    box = bounding_box(seg)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image=plot_box(image,box)

    #get the box range
    x_real, y_real = get_range(box,0,1.5)

    #show the boudning box
    if vis == True:
        
        #convert to color
        #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        #show box for sanity
        #img=plot_box(image,box)

        #draw pca line
        #cv2.line(image, (int(x[0]),int(y[0])), (int(x[1]),int(y[1])), (255,0,0), thickness=5, lineType=8, shift=0)
        
        #cv2.imwrite("/home/avengers/Desktop/centroid_locator/sample_output/"+str(count)+".jpg",img) 
        #show_box(image,box)
        #cv2.imshow("image",image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        pass
        
    #reset the intial guess for the next time step, doubling size on the way
    initial_guess_box = expand_box(box,2.0)
    
    image=plot_box(image,initial_guess_box)
    #if int(count) > 188:
    #    cv2.imshow("image",image)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    
    cv2.imwrite("/home/avengers/Desktop/centroid_locator/sample_output/"+str(count)+".jpg",image) 

    return initial_guess_box, x_real, y_real

def didget(number):

    #convert a counter number to a glob sortable number when images are in a folder

    #conver to string
    number=str(number)

    
    if len(number) == 1:
        frame="00"+number

    if len(number) == 2:
        frame="0"+number

    if len(number) == 3:
        frame=number
    
    return frame


def read(name,rot):

    #read an image and rotate if commanded
    
    #read the image
    image = cv2.imread(name,0)

    #convert to size
    image = cv2.resize(image,(1280,960))

    #rotate
    rows,cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
    image = cv2.warpAffine(image,M,(cols,rows))

    return image

def read_image_filenames(directory):

    #use glob to read in a folder of filenames

    #read
    names=sorted(glob.glob(directory+"*.jpg"))

    return names


def click_and_crop(event, x, y, flags, param):
    
    #select initial box

    global refPt, cropping

    #if clicked a point add it to the list
    if event == cv2.EVENT_LBUTTONDOWN and len(refPt) == 0:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONDOWN and len(refPt)!=0:
        refPt.append((x, y))
        cropping = False

        #draw up to 2 boxes
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.rectangle(image, refPt[2], refPt[3], (0, 255, 0), 2)

        #show the image
        cv2.imshow("image", image)



#read in the filenames
filenames=read_image_filenames("/home/avengers/Desktop/centroid_locator/ros bags/bag_1/")


#load the first image in the sequence
image = cv2.imread(filenames[0])
image = cv2.resize(image,(1280,960))
rows,cols,chan = image.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
image = cv2.warpAffine(image,M,(cols,rows))
image_backup = image
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
        global refPt
        refPt=[]
        print("RESET")

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# close all open windows
cv2.destroyAllWindows()


#eliminate the global varible
points=refPt

observations=[]

#if one object
if len(points) == 2:

    #init the search by selecting the area defined by the user
    initial_guess_box = init_box(points[:2],filenames)
    
    #initialize the counter for image saver
    count=0

    #loop through all filenames
    for name in filenames:

        #update the initial guess
        initial_guess_box, x_real, y_real = search_frame(initial_guess_box,name,didget(count),vis=True)

        observations.append(y_real)
        
        #update count
        count+=1

        if count > 125:
            #print observations
            pass
            #img = read(name,0)
            #show_box(img,initial_guess_box)

#if two objects
elif len(points)  == 4:

    #init the search by selecting the areas defined by the user
    initial_guess_box_1 = init_box(points[:2],filenames)
    initial_guess_box_2 = init_box(points[2:],filenames)

    #loop through the filenames
    for name in filenames:

        #update box intital guesses
        initial_guess_box_1 = search_frame(initial_guess_box_1,name,vis=False)
        initial_guess_box_2 = search_frame(initial_guess_box_2,name,vis=False)

        #show_boxes(img,initial_guess_box_1,initial_guess_box_2)
















