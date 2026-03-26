import cv2
import matplotlib.pyplot as plt
import os 
import numpy as np
from skimage.filters import gabor

def edge_extraction(image,x_i,y_i):
   edges_detected = cv2.Canny(image , x_i, y_i)
   return image,edges_detected



def get_all_jpg_pics(addr):
    jpg_files = []
    for path,name,files in os.walk(addr):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext == ".jpg" or ext == ".jpeg":
                jpg_files.append(os.path.join(path,file))
    return jpg_files



def detect_shapes(edges):
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=360)
    
    num_circles = 0 if circles is None else len(circles[0])

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=5, maxLineGap=10)
    
    num_lines = 0 if lines is None else len(lines)

    return num_circles, num_lines


def compute_gabor_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orientations = [0, 45, 90, 135] 
    frequency = 0.6  
    gabor_responses = []

    for theta in orientations:
        theta_rad = np.deg2rad(theta)
        
        real, _ = gabor(gray, frequency=frequency, theta=theta_rad)
        gabor_responses.append(real)


    gabor_mean_response = np.mean(gabor_responses, axis=0)
    texture_score = np.mean(gabor_mean_response)
    return texture_score

def assign_group(ts):
    if ts <= 903:
        return "FBC + PPI + SPC"
    elif 903 < ts <= 906:
        return "FBC + SPI + SPC"
    elif 906 < ts <= 909:
        return "FBC + Gluten + SPC"
def SME_load(t,t0,N,Nr,Pr,feed):
   effective_power = ((t-t0)/100) * (N/Nr)*Pr
   return effective_power/feed