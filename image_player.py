import cv2
import time

def show_image(image_path, duration=2):

    img = cv2.imread(image_path)

    if img is None:
        print("Image not found:", image_path)
        return

    cv2.imshow("Reaction", img)
    cv2.waitKey(int(duration * 1000))
    cv2.destroyWindow("Reaction")