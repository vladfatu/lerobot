import cv2
import numpy as np

def detect_circles(image):
    # Apply Gaussian blur to the image before converting to HSV
    blurred = cv2.GaussianBlur(image, (9, 9), 0)

    # Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define red color ranges in HSV
    lower_red1 = np.array([0, 130, 80])
    upper_red1 = np.array([5, 255, 255])

    lower_red2 = np.array([175, 130, 80])
    upper_red2 = np.array([180, 255, 255])

    # Threshold HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine both masks
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # First remove small noise
    red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Then fill small holes
    red_mask_clean = cv2.morphologyEx(red_mask_clean, cv2.MORPH_CLOSE, kernel)

    input_for_hough = cv2.GaussianBlur(red_mask_clean, (9, 9), 2)

    # Detect circles in the masked grayscale image
    circles = cv2.HoughCircles(
        input_for_hough,                         # Input image (masked grayscale)
        cv2.HOUGH_GRADIENT,                  # Detection method
        dp=1.2,                              # Accumulator resolution
        minDist=20,                          # Minimum distance between detected centers
        param1=100,                          # Higher threshold for Canny edge detector
        param2=20,                           # Accumulator threshold for circle detection
        minRadius=3,                         # Minimum circle radius
        maxRadius=100                        # Maximum circle radius
    )
    # Filter circles: keep only those where red_mask_clean[y, x] != 0
    if circles is None:
        return None, input_for_hough

    filtered = []
    N = 0.7  # 70% red pixels inside the circle
    for circle in circles[0, :]:
        x, y, r = np.uint16(np.around(circle))
        # Create a mask for the circle
        circle_mask = np.zeros_like(red_mask_clean, dtype=np.uint8)
        cv2.circle(circle_mask, (x, y), r, 255, -1)
        # Calculate the ratio of red_mask_clean > 0 inside the circle
        total_pixels = np.count_nonzero(circle_mask)
        if total_pixels == 0:
            continue
        red_pixels = np.count_nonzero((red_mask_clean > 0) & (circle_mask > 0))
        if red_pixels / total_pixels >= N:
            filtered.append(circle)

    # filter2: no circle in another circle
    # Sort circles by radius in descending order (prioritize large circles)
    filtered = sorted(filtered, key=lambda c: c[2], reverse=True)
    final_circles = []
    for c in filtered:
        x1, y1, r1 = c
        inside = False
        for c2 in final_circles:
            x2, y2, r2 = c2
            dist = np.hypot(x1 - x2, y1 - y2)
            if dist < r2:
                inside = True
                break
        if not inside:
            final_circles.append(c)
    filtered = final_circles

    return np.array(filtered), input_for_hough
