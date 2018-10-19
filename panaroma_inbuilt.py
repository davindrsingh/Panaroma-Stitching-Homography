'''The code to find matchings and keypoints is taken from "https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html"'''
import cv2
import numpy as np

img_1 = cv2.imread('img1.jpg',1)
img_2 = cv2.imread('img2.jpg',1)
img_3 = cv2.imread('img3.jpg',1)
img_4 = cv2.imread('img4.jpg',1)
img_2 = cv2.resize(img_2, (0, 0), fx=0.5, fy=0.5,interpolation=cv2.INTER_LANCZOS4)
img_3 = cv2.resize(img_3, (0, 0), fx=0.5, fy=0.5,interpolation=cv2.INTER_LANCZOS4)
img_4 = cv2.resize(img_4, (0, 0), fx=0.5, fy=0.5,interpolation=cv2.INTER_LANCZOS4)
img_1 = cv2.resize(img_1, (0, 0), fx=0.5, fy=0.5,interpolation=cv2.INTER_LANCZOS4)
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
       M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10)
       matchesMask = mask.ravel().tolist()
       h,w,k = img1.shape
       pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
       dst = cv2.perspectiveTransform(pts,M)
   else:
       print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
       matchesMask = None
   return M

def stitch_iamge(image1, image2, output):

    img_1 = image1
    img_2 = image2

    h = homography(img_1,img_2)

    h1, w1, d = img_1.shape
    h2, w2, d = img_2.shape
    print h2,w2
    new_ = output
    y_offset = 5000
    x_offset = 1000
    weight = []
    n = w2/2
    quant = 1.0/n
    for i in range(n):
        weight.append(0 + i*quant)
    weight.append(1)
    right = list(reversed(weight[:len(weight)-1]))
    weight+=right
    del right
    for i in range(h2):
        for j in range(w2):
            new_[i + x_offset][j + y_offset][0] = img_2[i][j][0]#*weight[j]
            new_[i + x_offset][j + y_offset][1] = img_2[i][j][1]#*weight[j]
            new_[i + x_offset][j + y_offset][2] = img_2[i][j][2]#*weight[j]
    for i in range(h1):
        for j in range(w1):
            y, x = i, j
            trans = np.dot(h, np.array([[x], [y], [1]]))
            trans = (1 / trans[2]) * trans
            v, u = trans[0], trans[1]
            #if new_[int(u) + x_offset][int(v) + y_offset][0] == 0 and new_[int(u) + x_offset][int(v) + y_offset][1] == 0 and new_[int(u) + x_offset][int(v) + y_offset][2] == 0:
            new_[int(u) + x_offset][int(v) + y_offset] = img_1[i][j]
            new_[int(u) + x_offset+1][int(v) + y_offset] = img_1[i][j]
            new_[int(u) + x_offset][int(v) + y_offset+1] = img_1[i][j]
            new_[int(u) + x_offset+1][int(v) + y_offset+1] = img_1[i][j]
            new_[int(u) + x_offset][int(v) + y_offset-1] = img_1[i][j]
            new_[int(u) + x_offset-1][int(v) + y_offset] = img_1[i][j]
            new_[int(u) + x_offset-1][int(v) + y_offset-1] = img_1[i][j]

    new_ = new_.astype("uint8")
    return new_
#########################################################
new_ = np.zeros((4000,8000,3),dtype='uint8')

#new_ = stitch_iamge(img_4,img_3,new_)

#cv2.imwrite("rev_new_out.jpg",new_)

h1 = homography(img_2,img_3) #for warping image two and three
h0 = homography(img_1,img_2)
h0_ = np.matmul(h0,h1) # for warping image one and two

x_offset = 1000
y_offset = 5000
x_min = np.matmul(h1,np.array([[0],[0],[1]]))
X_min = x_min/x_min[2]
weight = []
n = img_1.shape[1]/2
quant = 1.0/n
for i in range(n):
    weight.append(0 + i*quant)
weight.append(1)
l = [1]* (img_1.shape[1]-n)
weight+=l
del l
weight = list(reversed(weight))
for i in range(img_1.shape[0]):
    for j in range(img_1.shape[1]):
        y,x = i,j
        trans = np.matmul(h0_,np.array([[x],[y],[1]]))
        trans = (1 / trans[2]) * trans
        v,u = trans[0], trans[1]
        #if new_[int(u) + x_offset][int(v) + y_offset][0] == 0 or new_[int(u) + x_offset][int(v) + y_offset][1] == 0 or new_[int(u) + x_offset][int(v) + y_offset][2] == 0:
        new_[int(u) + x_offset][int(v) + y_offset] = img_1[i][j]*weight[j]
        new_[int(u) + x_offset + 1][int(v) + y_offset] = img_1[i][j]*weight[j]
        new_[int(u) + x_offset][int(v) + y_offset + 1] = img_1[i][j]*weight[j]
        new_[int(u) + x_offset + 1][int(v) + y_offset + 1] = img_1[i][j]*weight[j]
        new_[int(u) + x_offset][int(v) + y_offset - 1] = img_1[i][j]*weight[j]
        new_[int(u) + x_offset - 1][int(v) + y_offset] = img_1[i][j]*weight[j]
        new_[int(u) + x_offset - 1][int(v) + y_offset - 1] = img_1[i][j]*weight[j]


#cv2.imwrite("rev3_new_out_3.jpg",new_)

weight = []
n = img_1.shape[1]/2
quant = 1.0/n
for i in range(n):
    weight.append(0 + i*quant)
weight.append(1)
l = [1]* (img_1.shape[1]-n)
weight+=l
del l
weight = list(reversed(weight))
x_offset = 1000
y_offset = 5000
new_=  new_.astype('uint8')
for i in range(img_2.shape[0]):
    for j in range(img_2.shape[1]):
        y,x = i,j
        trans = np.matmul(h1,np.array([[x],[y],[1]]))
        trans = (1 / trans[2]) * trans
        v,u = trans[0], trans[1]
        #if new_[int(u) + x_offset][int(v) + y_offset][0] == 0 and new_[int(u) + x_offset][int(v) + y_offset][1] == 0 and new_[int(u) + x_offset][int(v) + y_offset][2] == 0:
        new_[int(u) + x_offset][int(v) + y_offset] = img_2[i][j]*weight[j]
        new_[int(u) + x_offset + 1][int(v) + y_offset] = img_2[i][j]*weight[j]
        new_[int(u) + x_offset][int(v) + y_offset + 1] = img_2[i][j]*weight[j]
        new_[int(u) + x_offset + 1][int(v) + y_offset + 1] = img_2[i][j]*weight[j]
        new_[int(u) + x_offset][int(v) + y_offset - 1] = img_2[i][j]*weight[j]
        new_[int(u) + x_offset - 1][int(v) + y_offset] = img_2[i][j]*weight[j]
        new_[int(u) + x_offset - 1][int(v) + y_offset - 1] = img_2[i][j]*weight[j]
new_ = new_.astype('uint8')
#cv2.imwrite("rev2_new_out.jpg",new_)
new_ = stitch_iamge(img_4,img_3,new_)
cv2.imwrite("inbuilt_panaroma.jpg",new_)