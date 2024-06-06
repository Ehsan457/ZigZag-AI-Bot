import cv2
import numpy as np


def nothing(x):
    pass


# Create a window with trackbars
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Thresh1", "Trackbars", 50, 500, nothing)
cv2.createTrackbar("Thresh2", "Trackbars", 100, 500, nothing)
cv2.createTrackbar("param1", "Trackbars", 50, 200, nothing)
cv2.createTrackbar("param2", "Trackbars", 30, 100, nothing)
cv2.createTrackbar("minRadius", "Trackbars", 10, 50, nothing)
cv2.createTrackbar("maxRadius", "Trackbars", 20, 100, nothing)

# Load your game screenshot
image_path = "path_to_image"  # Replace with your screenshot path
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

while True:
    # Get the current positions of the trackbars
    Thresh1 = cv2.getTrackbarPos("Thresh1", "Trackbars")
    Thresh2 = cv2.getTrackbarPos("Thresh2", "Trackbars")
    param1 = cv2.getTrackbarPos("param1", "Trackbars")
    param2 = cv2.getTrackbarPos("param2", "Trackbars")
    minRadius = cv2.getTrackbarPos("minRadius", "Trackbars")
    maxRadius = cv2.getTrackbarPos("maxRadius", "Trackbars")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, Thresh1, Thresh2)
    cv2.imshow("Canny Edges", edges)

    # Apply Hough Circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=20,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )

    # Draw the detected circles on the original image
    output = image.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow("Detected Circles", output)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
