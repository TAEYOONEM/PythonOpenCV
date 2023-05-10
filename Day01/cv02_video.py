import cv2

# 두대의 카메라를 사용하는 경우 Argument 0,1 두개의 객체 생성
# cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture("output.avi")
foutcc = cv2.VideoWriter_fourcc(*"XVID")
# writer = cv2.VideoWriter("output.avi",foutcc,30.0,(640,480))

# 영상 캡쳐를 반복
while(True) :
    ret,img_color = cap.read()

    # 캡쳐가 되지 않은 경우 첫줄부터 다시 실행
    if ret == False :
        # continue # 저장
        break # 저장된 파일 실행

    img_gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)

    cv2.imshow("Color", img_color)
    cv2.imshow("Gray", img_gray)

    # writer.write(img_color)

    # esc키 누르면 break
    if cv2.waitKey(1)& 0xff == 27 :
        break

cap.release()
# writer.release()
cv2.destroyAllWindows()
