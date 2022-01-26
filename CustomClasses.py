################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
##### Custom Classes used for the project are defined here.#####
################################################################

# Imports
import cv2
import math
from functools import reduce


# ImageObject class is used for saving the complete details of an input image.
# It contains the following information's
# 1. Image file for reference.
# 2. A List containing individual text regions detected from the image.
class ImageObject():
    def __init__(self, image, contour_list, image_name):
        self.image = image
        self.contour_list = contour_list
        self.image_name = image_name


# ContourObject class is used for saving the details of a  detected text region/contour region.
# It contains the following information's
# 1. Detected Contour feature.
# 2. x, y co-ordinate details.
# 3. Centroid of the detected contour region.
class ContourObject():
    def __init__(self, contour, x, y, text):
        # type: (object, object, object) -> object
        self.contour = contour
        self.x = x
        self.y = y
        self.text = text
        self.cX, self.cY = self.findCentroid(contour, x, y)

    def findCentroid(self, contour, x, y):

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
            print (x, y, "halo")
        # cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])
        return cx, cy


class TrainedData:
    def __init__(self, images, clusters):
        self.images = images
        self.clusters = clusters


class Point:
    def __init__(self, coords, text=None):
        self.coords = coords
        self.n = len(coords)
        self.text = text

    def __repr__(self):
        return str(self.coords)


class Cluster:
    def __init__(self, points):
        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        self.n = points[0].n

        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)

    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList) / numPoints for dList in unzipped]

        return Point(centroid_coords, self.points[0].text)


class Head:
    def __init__(self, contour, point):
        self.point = point
        self.cnt = contour


class Tail:
    def __init__(self, contour, point):
        self.point = point
        self.cnt = contour


class LineObject:
    def __init__(self, contour, left, right, top, bottom):
        self.lef = left
        self.rig = right
        self.top = top
        self.bot = bottom
        self.cnt = contour


class ArrowObject:
    def __init__(self, head, tail, line):
        self.head = head
        self.tail = tail
        self.line = line


def getDistance(a, b):
    '''
        Euclidean distance between two n-dimensional points.
        Note: This can be very slow and does not scale well
        '''
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")

    ret = reduce(lambda x, y: x + pow((a.coords[y] - b.coords[y]), 2), range(a.n), 0.0)
    return math.sqrt(ret)
