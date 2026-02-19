import cv2
import imageio
import time


def play_gif(gif_path):

    gif = imageio.mimread(gif_path)

    for frame in gif:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Emote GIF", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("Emote GIF")
