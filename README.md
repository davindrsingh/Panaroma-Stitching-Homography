# Panaroma-Stitching-Homography
Creating Panaroma using 4 images


homography.py is an implementaion to compute homography matrix between two images using RANSAC algorithm. The results are very much similar to the inbuilt CV function.

panaroma.py is an implementation of image stitching using homography matrix implementation. I have stitched four images together and also implemented some degrees of blending.

panaroma_inbuilt.py is an implementation of image stitching using the inbuilt homography function (cv2.findshomography()).

The results are stored in result folder.

Future Improvements - 
1. To perfect the blending.
2. To strip the redundant pixel from the canvas.
