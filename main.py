import cv2
import winsound
# open camera
cam = cv2.VideoCapture(0)

# Functions to do when camera is open/turned on
while cam.isOpened():
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()

    # So in order to detect the motion we need to have 2 frames taken at 2 diff instances
    # the diff between the frames gives us the movement/change
    diff = cv2.absdiff(frame1, frame2)

    # the abovediff is then grayscaled for easier readings
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # Convert the gray image into blurred image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create a threshold in order to get rid of noise
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Doing dilation which is just opposite of threshold
    dilated = cv2.dilate(thresh, None, iterations = 3)

    # border of the object/item detected
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
    for c in contours:
        if cv2.contourArea(c) < 20000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)
        winsound.PlaySound('LaughSOund.wav', winsound.SND_ASYNC)


    # if statement for when 'q' key is pressed the camera turns off
    if cv2.waitKey(1) == ord('q'):
        break

    cv2.imshow('Camera', frame1)

