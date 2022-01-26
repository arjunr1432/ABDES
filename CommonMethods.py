################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
##### Common Methods used for the project are defined here.#####
################################################################

# Imports
import math
from functools import reduce
from PIL import Image
import cv2
from PIL import ImageFilter
import pytesser
from CustomClasses import ContourObject, ImageObject, Point

image_file_dir_path = "images/diagrams/"
image_temp_dir_path = "images/temp/"
image_detected_dir_path = "images/detected/"
data_save_dir_path = "data/"


# Detected contour region details.
# arguments : (imageFileName)
# return : ImageObject


def processImage(path, input_image_name):
    image_file_path = path + input_image_name
    im = cv2.imread(image_file_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = Image.open(image_file_path)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    idx = 0
    contour_list = []
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        peri = cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        if len(approx) == 4:
            roi = im[y:y + h, x:x + w]
            file_name = image_temp_dir_path + str(idx) + '.jpg'
            cv2.imwrite(file_name, roi)

            img = Image.open(file_name)

            text = pytesser.image_to_string(img).rstrip('\n')
            text = text.replace('\n', ' ')
            text = text.strip(' ')
            if addCountourToList(contour_list, cnt, x, y, text):
                file_name = image_detected_dir_path + str(idx) + '.jpg'
                cv2.imwrite(file_name, roi)
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
                image = removeImageContent(image, [x - 3, y - 3, (x + w + 3), (y + h + 3)])
                print ("Text :", text, "$$")
    image.save(image_detected_dir_path + "Cut_" + input_image_name)
    imageObject = ImageObject(im, contour_list, input_image_name)
    return imageObject


# Method will create a ContourObject and add it to the contourList given in the argument.
# arguments : (contourList, cnt, x, y)
# return : void
def addCountourToList(contour_list, cnt, x, y, text):
    if len(contour_list) == 0 and text != "":
        contour_list.append(ContourObject(cnt, x, y, text))
        flag = True
    else:
        flag = True
        for contour_var in contour_list:
            tempX = contour_var.x - x
            tempY = contour_var.y - y
            if tempY < 5 and tempX < 5:
                print ("Point removed" + str(x) + str(y))
                flag = False
            if x == 1 and y == 1:
                flag = False
                print ("Point removed" + str(x) + str(y))
        if flag and (text != "" or len(text) != 0):
            contour_object = ContourObject(cnt, x, y, text)
            if contour_object.cX != 0 and contour_object.cY != 0:
                contour_list.append(contour_object)
                print (x, y, text)
            else:
                flag = False
        else:
            flag = False

    return flag


def removeImageContent(image, positions):
    image_crop_part = image.crop(positions)
    for i in range(100):  # You can blur many times
        image_crop_part = image_crop_part.filter(ImageFilter.BLUR)
    image.paste(image_crop_part, positions)
    return image


def generateInitialClusterPoints(images):
    init_contour_points = []
    init_contours_list = images[0].contour_list
    for contour in init_contours_list:
        init_contour_points.append(Point([contour.cX, contour.cY], contour.text))
    return init_contour_points


def generateAllContourPointsForClustering(images):
    contour_points = []
    for image in images:
        contour_list = image.contour_list
        for contour in contour_list:
            contour_points.append(Point([contour.cX, contour.cY], contour.text))
    return contour_points


def getDistance(a, b):
    """
        Euclidean distance between two n-dimensional points.
        Note: This can be very slow and does not scale well
        """
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")

    ret = reduce(lambda x, y: x + pow((a.coords[y] - b.coords[y]), 2), range(a.n), 0.0)
    return math.sqrt(ret)
