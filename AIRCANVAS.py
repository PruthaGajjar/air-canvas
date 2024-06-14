from HandTrack import *
import cv2
import numpy as np
import random
import os

os.environ["TF_LOG"] = "warning"  # Hide INFO and lower level logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class ColorRect():
    def __init__(self, x, y, w, h, color, text='', alpha=0.5, image=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.alpha = alpha
        self.image = image

    def drawRect(self, img, text_color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        if self.alpha is None:
            self.alpha = 0.5
        bg_rec = img[self.y: self.y + self.h, self.x: self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        if isinstance(self.alpha, np.ndarray):
            if self.alpha.size == 1:
                alpha = max(0, min(self.alpha.item(), 1))  # Convert array to scalar using item()
            else:
                # Handle array case, take mean value
                alpha = max(0, min(np.mean(self.alpha), 1))
        else:
            alpha = max(0, min(self.alpha, 1))

        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1 - alpha, 1.0)

        # Putting the image back to its position
        img[self.y: self.y + self.h, self.x: self.x + self.w] = res

        # put the letter
        tetx_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w / 2 - tetx_size[0][0] / 2), int(self.y + self.h / 2 + tetx_size[0][1] / 2))
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def isOver(self, x, y):
        if (self.x + self.w > x > self.x) and (self.y + self.h > y > self.y):
            return True
        return False


# Function to draw shapes based on user selection
def draw_shape(frame, shape, start_point, end_point, color, thickness):
    if shape == "line":
        cv2.line(frame, start_point, end_point, color, thickness)
    elif shape == "rectangle":
        cv2.rectangle(frame, start_point, end_point, color, thickness)
    elif shape == "circle":
        radius = int(((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2) ** 0.5)
        cv2.circle(frame, start_point, radius, color, thickness)


# initilize the hand detector
detector = HandTracker(detectionCon=0.80)  # Change integer percentage to float value

# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# initilize the camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# creating canvas to draw on it
canvas = np.zeros((720, 1280, 3), np.uint8)

# define a previous point to be used with drawing a line
px, py = 0, 0
# initial brush color
color = (255, 0, 0)
#####
brushSize = 5
eraserSize = 20
####

########### creating colors ########
# Colors button
colorsBtn = ColorRect(200, 0, 100, 100, (120, 255, 0), 'Colors')

colors = []
# random color
b = int(random.random() * 255) - 1
g = int(random.random() * 255)
r = int(random.random() * 255)
print(b, g, r)
colors.append(ColorRect(300, 0, 100, 100, (b, g, r)))
# red
colors.append(ColorRect(400, 0, 100, 100, (0, 0, 255)))
# blue
colors.append(ColorRect(500, 0, 100, 100, (255, 0, 0)))
# green
colors.append(ColorRect(600, 0, 100, 100, (0, 255, 0)))
# yellow
colors.append(ColorRect(700, 0, 100, 100, (0, 255, 255)))
# erase (black)
colors.append(ColorRect(800, 0, 100, 100, (0, 0, 0), "Eraser"))

# clear
clear = ColorRect(900, 0, 100, 100, (100, 100, 100), "Clear")

# Load shape images
line_img = cv2.imread('line_image.png')
rectangle_img = cv2.imread('circle.png')
circle_img = cv2.imread('rectangle.png')

# shapes buttons with images
shapes = [
    ColorRect(1160, 50 + 100 * i, 100, 100, (50, 50, 50), str(shape), image)
    for i, (shape, image) in enumerate(zip(['Line', 'Rect', 'Circle'], [line_img, rectangle_img, circle_img]))
]

# ########## SHAPES #######
# shapes = []
# for i, shape in enumerate(range(5, 25, 5)):
#     shapes.append(ColorRect(1160, 50 + 100 * i, 100, 100, (50, 50, 50), str(shape)))
# shape button
shapesBtn = ColorRect(1160, 0, 100, 50, color, 'Shapes')

########## pen sizes #######
pens = []
for i, penSize in enumerate(range(5, 25, 5)):
    pens.append(ColorRect(1040, 50 + 100 * i, 100, 100, (50, 50, 50), str(penSize)))

penBtn = ColorRect(1040, 0, 100, 50, color, 'Pen')

# white board button
boardBtn = ColorRect(50, 0, 100, 100, (255, 255, 0), 'Board')

# define a white board to draw on
whiteBoard = ColorRect(10, 120, 1020, 580, (255, 255, 255), alpha=0.6)

coolingCounter = 60
hideBoard = True
hideColors = True
hidePenSizes = True
hideShapes = True

while True:

    if coolingCounter:
        coolingCounter -= 1
        # print(coolingCounter)

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)

    detector.findHands(frame)
    positions = detector.getPosition(frame, draw=False)
    upFingers = detector.getUpFingers(frame)

    if upFingers:
        x, y = positions[8][0], positions[8][1]
        if upFingers[1] and not whiteBoard.isOver(x, y):
            px, py = 0, 0

            ##### pen sizes ######
            if not hidePenSizes:
                for pen in pens:
                    if pen.isOver(x, y):
                        brushSize = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5

            ####### chose a shape for drawing #######
            if not hideShapes:
                for sh in shapes:
                    if sh.isOver(x, y):
                        color = sh.color
                        sh.alpha = 0
                    else:
                        sh.alpha = 0.5

            ####### chose a color for drawing #######
            if not hideColors:
                for cb in colors:
                    if cb.isOver(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

                # Clear
                if clear.isOver(x, y):
                    clear.alpha = 0
                    canvas = np.zeros((720, 1280, 3), np.uint8)
                else:
                    clear.alpha = 0.5

            # color button
            if colorsBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                colorsBtn.alpha = 0
                hideColors = False if hideColors else True
                colorsBtn.text = 'Colors' if hideColors else 'Hide'
            else:
                colorsBtn.alpha = 0.5

            # Pen size button
            if penBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                penBtn.alpha = 0
                hidePenSizes = False if hidePenSizes else True
                penBtn.text = 'Pen' if hidePenSizes else 'Hide'
            else:
                penBtn.alpha = 0.5

            # shapesBtn
            if shapesBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                hideShapes = False if hideShapes else True
                shapesBtn.text = 'Shapes' if hideShapes else 'Hide'
            else:
                # Check if any shape button is clicked
                if not hideShapes:
                    for sh in shapes:

                        if sh.isOver(x, y):
                            if sh.text == 'Line':
                                # Draw a horizontal line on the whiteboard
                                cv2.line(canvas, (110, 250), (410, 250), color, brushSize)
                            elif sh.text == 'Rect':
                                # Draw a smaller rectangle on the whiteboard
                                cv2.rectangle(canvas, (260, 250), (660, 400), color, brushSize)
                            elif sh.text == 'Circle':
                                # Draw a smaller circle on the whiteboard
                                cv2.circle(canvas, (480, 320), 50, color, brushSize)

            # white board button
            if boardBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                boardBtn.alpha = 0
                hideBoard = False if hideBoard else True
                boardBtn.text = 'Board' if hideBoard else 'Hide'

            else:
                boardBtn.alpha = 0.5

        elif upFingers[1] and not upFingers[2]:
            if whiteBoard.isOver(x, y) and not hideBoard:
                # print('index finger is up')
                cv2.circle(frame, positions[8], brushSize, color, -1)
                # drawing on the canvas
                if px == 0 and py == 0:
                    px, py = positions[8]
                if color == (0, 0, 0):
                    cv2.line(canvas, (px, py), positions[8], color, eraserSize)
                else:
                    cv2.line(canvas, (px, py), positions[8], color, brushSize)
                px, py = positions[8]

        else:
            px, py = 0, 0

    # put colors button
    colorsBtn.drawRect(frame)
    cv2.rectangle(frame, (colorsBtn.x, colorsBtn.y), (colorsBtn.x + colorsBtn.w, colorsBtn.y + colorsBtn.h),
                  (255, 255, 255), 2)

    # put white board buttin
    boardBtn.drawRect(frame)
    cv2.rectangle(frame, (boardBtn.x, boardBtn.y), (boardBtn.x + boardBtn.w, boardBtn.y + boardBtn.h), (255, 255, 255),
                  2)
    # shapes
    shapesBtn.color = color
    shapesBtn.drawRect(frame)
    cv2.rectangle(frame, (shapesBtn.x, shapesBtn.y), (shapesBtn.x + shapesBtn.w, shapesBtn.y + shapesBtn.h),
                  (255, 255, 255), 2)

    # put the white board on the frame
    if not hideBoard:
        whiteBoard.drawRect(frame)
        ########### moving the draw to the main image #########
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)

    ########## pen colors' boxes #########
    if not hideColors:
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)

        clear.drawRect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (255, 255, 255), 2)

    ########## shapes boxes #########

    if not hideShapes:
        for sh in shapes:
            sh.drawRect(frame)
            cv2.rectangle(frame, (sh.x, sh.y), (sh.x + sh.w, sh.y + sh.h), (255, 255, 255), 2)

    ########## brush size boxes ######
    penBtn.color = color
    penBtn.drawRect(frame)
    cv2.rectangle(frame, (penBtn.x, penBtn.y), (penBtn.x + penBtn.w, penBtn.y + penBtn.h), (255, 255, 255), 2)
    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)
            cv2.rectangle(frame, (pen.x, pen.y), (pen.x + pen.w, pen.y + pen.h), (255, 255, 255), 2)

    cv2.imshow('video', frame)
    # cv2.imshow('canvas', canvas)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
