
# Program To Read video 
# and Extract Frames 
import cv2 


def didget(number):

    number=str(number)

    if len(number) == 1:
        frame="00"+number

    if len(number) == 2:
        frame="0"+number

    if len(number) == 3:
        frame=number
    
    return frame

# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        frame=didget(count)
        cv2.imwrite("test_images_2/"+frame+".jpg", image) 
  
        count += 1
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("/home/avengers/Desktop/centroid_locator/test_video_2.MOV") 
