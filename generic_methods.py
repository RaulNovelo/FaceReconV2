import cv2

# BGR color especifications
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def convertToGray(img):
    """Returns a gray scale version of given img. Used when detecting faces"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def drawRectangleText(img, rect_coords, rect_color, text, text_color):
    """Draw a rectangle with the given coordinates (rect) in the image"""
    x, y, w, h = rect_coords

    cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 2)
    if not text=="":
        cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, text_color, 2)

    return img