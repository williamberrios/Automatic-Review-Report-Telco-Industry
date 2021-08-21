from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

class detector:
    def __init__(self,filename):
        self.MAX_FEATURES = 500
        self.GOOD_MATCH_PERCENT = 0.15
        self.filename  = filename
        
        #load reference image
        self.img_reference  = cv2.imread(self.filename)
        
    def equalization(self,img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    def brightness(self,img):
        if len(img.shape) == 3:
            # Colored RGB or BGR (*Do Not* use HSV images with this function)
            # create brightness with euclidean norm
            return np.average(norm(img, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            return np.average(img)
    def adjust_gamma(self,image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def fix_brightness(self,img,equa=True):
        if self.brightness(img)<91:
            img = self.adjust_gamma(img,1.45)
        else:
            img = img.copy()    
        if equa:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img[:,:,0]= gray
            img[:,:,1]= gray
            img[:,:,2]= gray
            img = self.equalization(img)
        return img

    def predict(self,im1,show=False):
        #im1 = self.fix_brightness(im1,equa=False)
        im2 = self.img_reference
        im1 = im1.copy()
        im2 = im2.copy()
        
        # Relative resize
        if im1.shape[0]<700:
            im2 = cv2.resize(im2,(int(700*0.7),int(1000*0.7)))

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)


        if show:
            plt.imshow(imMatches)
            plt.show()
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))
        
        # Get fecha and firmas
        if im1.shape[0]<700:
            fecha = im1Reg[int(75*0.7):int(108*0.7),int(380*0.7):int(550*0.7),:]
            firmas = im1Reg[int(900*0.7):,:,:]
        else:
            fecha = im1Reg[75:108,380:550,:]
            firmas = im1Reg[900:,:,:]
            
        # resize images
        fecha = cv2.resize(fecha,(170,25))
        firmas = cv2.resize(firmas,(700,100))
        
        # split firmas
        firma1 = firmas[:,:350,:]
        firma2 = firmas[:,350:,:]
        
        fecha  = self.fix_brightness(fecha,equa=False)
        firmas = self.fix_brightness(firmas,equa=False)
        firma1 = self.fix_brightness(firma1,equa=False)
        firma2 = self.fix_brightness(firma2,equa=False)

        return im1Reg,fecha,firma1,firma2
