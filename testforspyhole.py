from __future__ import division
import math
import cv2
import numpy as np
from PIL import Image

def straighten_horizon(im, lhs, rhs):
    """
    Straighten the horizon of the image by skewing either end towards the centre of the
    difference between the two
    :param im: A PIL image
    :param lhs: Horizon at the left-hand end
    :param rhs: Horizon at the right hand end
    :return: A straightened PIL image
    """
    diff = (lhs - rhs) / 2
    imgX, imgY = im.size

    a = (diff / (imgX / 2)) * -1

    def yoffset(x):
        """
        Linear function - f(x) = ax + b
        """
        return int((a * x) + diff)

    # Convert image to numpy array for faster pixel manipulation
    im_array = np.array(im)

    for x in range(0, imgX // 2):
        dy = yoffset(x)
        im_array[:, x] = np.roll(im_array[:, x], dy, axis=0)
        im_array[:, imgX - x - 1] = np.roll(im_array[:, imgX - x - 1], -dy, axis=0)

    return Image.fromarray(im_array)

def equirectangular_to_tiny_planet(frame, lhs=2000, rhs=1400):
    # Convert OpenCV image (BGR) to PIL image (RGB)
    linear_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Straighten the horizon
    linear_image = straighten_horizon(linear_image, lhs, rhs)
    linear_image = linear_image.transpose(Image.FLIP_TOP_BOTTOM)

    imgX, imgY = linear_image.size

    # Create the output image
    circle_image = Image.new("RGBA", (imgX, imgY), (255, 255, 255, 0))

    # Rectangle to polar coordinates
    maxradius = imgX / 2
    rscale = imgX / maxradius
    tscale = imgY / (2 * math.pi)

    for y in range(0, imgY):
        dy = y - imgY // 2
        for x in range(0, imgX):
            dx = x - imgX // 2
            t = math.atan2(dy, dx) % (2 * math.pi) * tscale
            r = math.sqrt(dx**2 + dy**2) * rscale

            if 0 <= t < imgX and 0 <= r < imgY:
                circle_image.putpixel((x, y), linear_image.getpixel((int(t), int(r))))

    # Convert PIL image back to OpenCV image
    return cv2.cvtColor(np.array(circle_image), cv2.COLOR_RGBA2BGR)
    # return cv2.cvtColor(np.array(linear_image), cv2.COLOR_RGBA2BGR)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set the frame rate
    frame_rate = 30
    prev = 0

    while True:
        time_elapsed = cv2.getTickCount() / cv2.getTickFrequency() - prev

        if time_elapsed > 1.0 / frame_rate:
            prev = cv2.getTickCount() / cv2.getTickFrequency()

            ret, frame = cap.read()

            if not ret:
                break

            # Convert frame to tiny planet effect
            tiny_planet_frame = equirectangular_to_tiny_planet(frame)

            cv2.imshow('Tiny Planet Effect', tiny_planet_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()