import cv2

imageFiles = ['./stitching/boat1.jpg','./stitching/boat2.jpg']
images = []
for filename in imageFiles:
    img = cv2.imread(filename)
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    images.append(img)

stitcher = cv2.createStitcher()
status, result = stitcher.stitch(images)             

cv2.imshow("Panorama", result)
cv2.waitKey()
cv2.destroyAllWindows()

