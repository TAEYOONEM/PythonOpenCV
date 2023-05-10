import cv2

img_color = cv2.imread('ex.jpg',cv2.IMREAD_COLOR)

# cv2.namedWindow('Show Image')
cv2.imshow('Show Image',img_color)
cv2.waitKey(0)


img_gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
cv2.imshow('Show GrayScaleImage',img_gray)
cv2.waitKey(0)

cv2.imwrite("savedImage.jpg",img_gray)
cv2.destroyAllWindows()