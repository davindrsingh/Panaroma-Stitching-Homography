''' Author - Davinder Singh'''
'''The code to find matchings and keypoints is taken from "https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html"'''
import numpy as np
import cv2
import random

# DLT method to get homography matrix
def matrix(P,Q):
   A = np.zeros((8,9))
   for i in range(4):
       x,y = P[i]
       u,v = Q[i]
       A[2*i] = [x,y,1,0,0,0,-x*u,-y*u,-u]
       A[2*i+1] = [0,0,0,x,y,1,-x*v,-y*v,-v]

   U, S, V = np.linalg.svd(A, full_matrices=False)
   A_ = V[-1]
   H = np.reshape(A_, (3, 3))
   H = H/H.item(8)
   return H


def RANSAC(src_pts, dst_pts, n, tolerance): #n = number of iterations
    itr_inliers = {} #store inliers for each iteration
    itr_homos = {} #store homography matrix for each iteration and then later choose the best one
    for j in range(n):
        indices = random.sample(range(len(src_pts)), 4) #pick four point at random
        P = []
        Q = []
        for index in indices:
            P.append((src_pts[index][0][0],src_pts[index][0][1]))
            Q.append((dst_pts[index][0][0], dst_pts[index][0][1]))
        estimate = matrix(P,Q) # generate homography matrix using DLT
        inliers = [] #contains src_pts , dst_pts pairs which are inliers
        #calculating inliers
        for i in range(len(src_pts)):
            X = np.array([[int(src_pts[i][0][0])],[int(src_pts[i][0][1])],[1]])
            V_estimate = np.dot(estimate,X)
            V_estimate = (1/V_estimate[2])*V_estimate

            V = np.array([[dst_pts[i][0][0]],[dst_pts[i][0][1]],[1]])

            if np.linalg.norm(V_estimate-V)<=tolerance:
                inliers.append([src_pts[i],dst_pts[i]])
        itr_inliers[j] = inliers
        itr_homos[j] = estimate
    T = 0.95 #threshold for percentage of inliers
    total = len(src_pts)
    max = -1
    final_id = 0
    for id in itr_inliers.keys():
        num_inl = len(itr_inliers[id])
        perc_inliers = num_inl/float(total)
        if perc_inliers>=T:
            final_id = id
            break #break out of the for loop
        else:
            if perc_inliers>max:
                final_id = id
                max = perc_inliers
    final_inliers = itr_inliers[final_id]
    if len(final_inliers)<4: #If number of max inliers is less than 4 than return homography matrix direclty.
                            # It should be noted that the estimate in this case would not be reliable
        return None, None
    elif len(final_inliers)%2==0:
        rn = len(final_inliers)
    else:
        rn = len(final_inliers)-1

    A_final = np.zeros((2*rn, 9))
    for k in range(rn):
        x,y = final_inliers[k][0][0][0],final_inliers[k][0][0][1]
        u,v = final_inliers[k][1][0][0],final_inliers[k][1][0][1]
        A_final[2 * k] = [x, y, 1, 0, 0, 0, -x * u, -y * u, -u]
        A_final[2 * k + 1] = [0, 0, 0, x, y, 1, -x * v, -y * v, -v]
    U, S, V = np.linalg.svd(A_final, full_matrices=False)
    A_ = V[-1]
    H = np.reshape(A_, (3, 3))
    H = H/H.item(8)  # Normalize
    return H,len(final_inliers)

def homography(img1,img2):
   sift = cv2.xfeatures2d.SIFT_create()
   kp1, des1 = sift.detectAndCompute(img1,None)
   kp2, des2 = sift.detectAndCompute(img2,None)

   FLANN_INDEX_KDTREE = 0
   index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   search_params = dict(checks = 50)
   flann = cv2.FlannBasedMatcher(index_params, search_params)
   matches = flann.knnMatch(des1,des2,k=2)
   # store all the good matches as per Lowe's ratio test.
   good = []
   for m,n in matches:
       if m.distance < 0.7*n.distance:
           good.append(m)
   MIN_MATCH_COUNT = 10
   if len(good)>MIN_MATCH_COUNT:
       src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
       dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
       flag = None #the RANSAC function returns NONE in case the key points are very low(<4), in such case we run the Function once again to get better estimate
       while(flag==None):
           try:
               M, I = RANSAC(src_pts, dst_pts, 5000, 5)
               flag=I
           except:
               #try again key points are very low in case key points are very low
               #helps to get a good estimate of homography matrix
               flag=1
   else:
       print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
return M
