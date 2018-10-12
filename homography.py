import cv2
import numpy as np

images = {}
for i in range(1,7):
    images[i] = cv2.imread("img"+str(i)+".JPG",0)

img_1 = cv2.imread('img1.JPG',cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread('img2.JPG',cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(img_1,None)
kp2,des2 = sift.detectAndCompute(img_2,None)

img_1 = cv2.drawKeypoints(img_1,kp1,None)
img_2 = cv2.drawKeypoints(img_2,kp2,None)

cv2.imwrite('sift_keypoints_1.jpg',img_1)
cv2.imwrite('sift_keypoints_2.jpg',img_2)

bf = cv2.BFMatcher_create(cv2.NORM_L2,crossCheck=False)
matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)
matching_resulst = cv2.drawMatches(img_1,kp1,img_2,kp2,matches[:100],None)

cv2.imwrite("result.jpg",matching_resulst)

# getting keypoints
kyp_1 = []
kyp_2 = []
for i in range(100):
    ind1,ind2 = matches[i].trainIdx, matches[i].queryIdx
    kyp_1.append(kp1[ind2].pt)
    kyp_2.append(kp2[ind1].pt)


# Getting Homography Matrix

#h, status = cv2.findHomography(pts_src, pts_dst) #inbuilt function for reference


# Using DLT Algorithm
def homography(P,Q):
    A = np.zeros((8,9))
    for i in range(4):
        x,y = P[i]
        u,v = Q[i]
        #A[2*i] = [-x,-y,-1,0,0,0,u*x,u*y,u]
        #A[2*i+1] = [0,0,0,-x,-y,-1,x*v,y*x,v]
        A[2*i] = [x,y,1,0,0,0,-u*x,-u*y,-u]
        A[2*i+1] = [0,0,0,x,y,1,x*v,y*x,v]

    [U, S, V] = np.linalg.svd(A)
    A_ = V[-1, :]
    H = np.reshape(A_, (3, 3))
    return H

P = [(int(pt[0]),int(pt[1])) for pt in kyp_1[:4]]
Q = [(int(pt[0]),int(pt[1])) for pt in kyp_2[:4]]
print P,Q
print homography(P,Q)
