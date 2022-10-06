from .models import *
import numpy as np
import cv2

class PositionCalculator:
    def calculate_middle_point(self, rectangle1: Rectangle, rectangle2: Rectangle):
        centerX1 = (rectangle1.topLeft.x + rectangle1.bottomRight.x)/2
        centerY1 = (rectangle1.topLeft.y + rectangle1.bottomRight.y)/2
        centerX2 = (rectangle2.topLeft.x + rectangle2.bottomRight.x)/2
        centerY2 = (rectangle2.topLeft.y + rectangle2.bottomRight.y)/2
        cX = (centerX1 + centerX2)/2
        cY = (centerY1 + centerY2)/2
        return Point(cX, cY)
    
    def calculate_angle_degrees(self, rvec1, rvec2):
        R, _ = cv2.Rodrigues(rvec1)
        # convert (np.matrix(R).T) matrix to array using np.squeeze(np.asarray()) to get rid off the ValueError: shapes (1,3) and (1,3) not aligned
        R = np.squeeze(np.asarray(np.matrix(R).T))
        val1 = R[2]
        
        R, _ = cv2.Rodrigues(rvec2)
        # convert (np.matrix(R).T) matrix to array using np.squeeze(np.asarray()) to get rid off the ValueError: shapes (1,3) and (1,3) not aligned
        R = np.squeeze(np.asarray(np.matrix(R).T))
        val2 = R[2]
        
        angle_radians = np.arccos(np.dot(val1, val2))
        angle_degrees = angle_radians*180/np.pi
        return angle_degrees

    def inversePerspective(self, rvec, tvec):
        """ Applies perspective transform for given rvec and tvec. """
        R, _ = cv2.Rodrigues(rvec)
        R = np.matrix(R).T
        invTvec = np.dot(R, np.matrix(-tvec))
        invRvec, _ = cv2.Rodrigues(R)
        return invRvec, invTvec


    def relativePosition(self, rvec1, tvec1, rvec2, tvec2):
        """ Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 """
        rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
        rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

        # Inverse the second marker, the right one in the image
        invRvec, invTvec = self.inversePerspective(rvec2, tvec2)

        info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
        composedRvec, composedTvec = info[0], info[1]

        composedRvec = composedRvec.reshape((3, 1))
        composedTvec = composedTvec.reshape((3, 1))
        return composedRvec, composedTvec