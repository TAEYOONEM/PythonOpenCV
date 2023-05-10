import cv2

def nothing(x) :
    pass

cv2.namedWindow('Binary')
cv2.createTrackbar('threshold', 'Binary', 0, 255, nothing)
cv2.setTrackbarPos('threshold', 'Binary', 127)

img_color = cv2.imread("eminemblack.jpg",cv2.IMREAD_COLOR)
# cv2.imshow("Color",img_color)
# cv2.waitKey(0)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Gray",img_gray)
# cv2.waitKey(0)
while True :
    low = cv2.getTrackbarPos('threshold', 'Binary')

    # threshold(대상이미지, 쓰레스홀드(기준), 255,THRESH_BINARY)
    # 마지막 argument가 THRESH_BINARY 일떄 쓰레스홀드보다 크면 3번쨰 Argument로
    # 작으면 0
    ret,img_binary = cv2.threshold(img_gray, low, 255, cv2.THRESH_BINARY_INV) 
    cv2.imshow("Binary",img_binary)

    img_result = cv2.bitwise_and(img_color, img_color, mask=img_binary)
    cv2.imshow("result",img_result)
    # cv2.waitKey(0)
    if cv2.waitKey(1)&0xFF == 27 :
        break

cv2.destroyAllWindows()

