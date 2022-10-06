'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


from typing import List, Optional
import numpy as np
import cv2
import sys
from numpy.lib.arraysetops import isin
import math
from numpy.lib.npyio import save
from helpers import *
from utils import ARUCO_DICT
import argparse
import time
from scipy.signal import lfilter, savgol_filter
from dataclasses import dataclass
from datetime import datetime, timedelta
#id : size in meters for Anton's markers for robot hand
markers_size = {0: 3/100, # lower robot part
                1: 2/100, # near marker robot part
                2: 2.5/100, # top robot part, connection of lower and top parts
                3: 5/100, # floor
                4: 4/100} # top robot part, back side, sticker that faces ceiling
measure_marker_pairs = [
    [0, 2],
    # [3, 1],
    # [2, 1],
    # [4, 1],
    # [4, 2],
    # [0, 3]
]
def marker_pair_id(pair: List[int]):
    return f'{pair[0]}_{pair[1]}'

         
class Outliers:
    def __init__(self, id: str) -> None:
        self.max_deviations = 1.5
        self.values = []
        self.values_limit = 100
        self.id = id
    
    def process_value(self, val):
        if val is not None:
            self.insert_value(val)
        if len(self.values) == self.values_limit:
            self.remove_last_value()
            
    def insert_value(self, val):
        self.values.insert(0, val)
    
    def remove_last_value(self):
        del self.values[-1]
        
    def filter_outliers(self):
        if len(self.values) <= 100:
            return self.values
        
        return savgol_filter(self.values, 99, 2)


    
class MarkerStorage:
    def __init__(self, markers: List[Marker]) -> None:
        self.markers = markers
    def get_marker(self, id: int) -> Optional[Marker]:
        for marker in self.markers:
            if marker.id == id:
                return marker
        return None
    def get_markers(self):
        return self.markers
    
    def set_marker_position(self, marker: Marker, topLeft, topRight, bottomRight, bottomLeft):
        cached_marker = self.get_marker(marker.id)
        if cached_marker:
            cached_marker.rectangle = Rectangle(topLeft = Point(topLeft), topRight = Point(topRight),
                                     bottomRight = Point(bottomRight), bottomLeft = Point(bottomLeft))
        else:
            raise Exception("Error marker is not found in cache")

class OutliersProcessor:
    def __init__(self) -> None:
        # create array of outlier filters for each pair of markers
        self.outlier_filters_degree = [Outliers(id=marker_pair_id(pair)) for pair in measure_marker_pairs]
        self.outlier_filters_distance = [Outliers(id=marker_pair_id(pair)) for pair in measure_marker_pairs]
    
    # processes degree, removes outliers and returns the last value
    def processed_degree(self, pair: List, degree: float):
        pair_id = marker_pair_id(pair)
        outlier_filter_deg = [filter for filter in self.outlier_filters_degree if filter.id == pair_id][0]
        outlier_filter_deg.process_value(degree)
        filtered = outlier_filter_deg.filter_outliers()[0]
        return filtered
    
    def processed_distance(self, pair: List, distance: float):
        pair_id = marker_pair_id(pair)
        outlier_filter_distance = [filter for filter in self.outlier_filters_distance if filter.id == pair_id][0]
        outlier_filter_distance.process_value(distance)
        filtered = outlier_filter_distance.filter_outliers()[0]
        return filtered
    
class DistanceMeasure:
    def __init__(self) -> None:

        self.CACHED_PTS = None
        self.CACHED_IDS = None
        self.Line_Pts = None
        self.measure = None

        self.Dist = []
    def measure_distance(self, image, corners, ids, marker_storage: MarkerStorage, degree_processor: any, distance_processor:any,
                         matrix_coefficients, distortion_coefficients):
        
        if len(corners) <= 0:
            if self.CACHED_PTS is not None:
                corners = self.CACHED_PTS
        if len(corners) > 0:
            print(f"[Info] {len(corners)} corners detected")
            self.CACHED_PTS = corners
            if ids is not None:
                ids = ids.flatten()
                self.CACHED_IDS = ids
            else:
                if self.CACHED_IDS is not None:
                    ids = self.CACHED_IDS
            if len(corners) < 2:
                if len(self.CACHED_PTS) >= 2:
                    corners = self.CACHED_PTS
            
        # draw markers
        
        for pair in measure_marker_pairs:
            first_marker_id = pair[0]
            second_marker_id = pair[1]
            
            first_marker = marker_storage.get_marker(first_marker_id)
            if first_marker is None:
                print('[Error] first marker is none')
                continue
            second_marker = marker_storage.get_marker(second_marker_id)
            if second_marker is None:
                print('[Error] second_marker marker is none')
                continue
            
            # contains center X, center Y of rectangles
            self.Dist = []
            for marker in [first_marker, second_marker]:
                markerCorner = marker.corners
                markerId = marker.id
                if markerId not in [first_marker_id, second_marker_id]:
                    continue
                
                corners_abcd = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners_abcd
                marker = marker_storage.get_marker(markerId)
                if marker is None:
                    print(f'[Warning] - marker with id {markerId} is not found')
                    continue
                marker_storage.set_marker_position(marker, topLeft, topRight, bottomRight, bottomLeft)
                
                topRightPoint = (int(topRight[0]), int(topRight[1]))
                topLeftPoint = (int(topLeft[0]), int(topLeft[1]))
                bottomRightPoint = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeftPoint = (int(bottomLeft[0]), int(bottomLeft[1]))
                cv2.line(image, topLeftPoint, topRightPoint, (0, 255, 0), 2)
                cv2.line(image, topRightPoint, bottomRightPoint, (0, 255, 0), 2)
                cv2.line(image, bottomRightPoint, bottomLeftPoint, (0, 255, 0), 2)
                cv2.line(image, bottomLeftPoint, topLeftPoint, (0, 255, 0), 2)
                cX = int((topLeft[0] + bottomRight[0])//2)
                cY = int((topLeft[1] + bottomRight[1])//2)
                
                def put_text(text, pos: Point, textColor):
                    pos.x = int(pos.x)
                    pos.y = int(pos.y)
                    cv2.putText(image, str(text), (pos.x, pos.y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,),3)
                    cv2.putText(image, str(text), (pos.x, pos.y), cv2.FONT_HERSHEY_COMPLEX, 1, textColor, 2)
                    
                cv2.circle(image, (cX, cY), 4, (255, 0, 0), -1)
                put_text(markerId, Point(int(topLeft[0]-10), int(topLeft[1]-10)), (0,0,255))
                
                self.Dist.append((cX, cY))
                
                if len(self.Dist) == 0:
                    if self.Line_Pts is not None:
                        self.Dist = self.Line_Pts
                if len(self.Dist) == 2:
                    self.Line_Pts = self.Dist
                    
                    firstRvec = first_marker.rvec
                    firstTvec = first_marker.tvec
                    secondRvec = second_marker.rvec
                    secondTvec = second_marker.tvec
                    firstRvec, firstTvec = firstRvec.reshape((3, 1)), firstTvec.reshape((3, 1))
                    secondRvec, secondTvec = secondRvec.reshape((3, 1)), secondTvec.reshape((3, 1))

                    composedRvec, composedTvec = PositionCalculator().relativePosition(firstRvec, firstTvec, secondRvec, secondTvec)
                    if composedRvec is not None and composedTvec is not None:
                        axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
                        info = cv2.composeRT(composedRvec, composedTvec, secondRvec, secondTvec)
                        TcomposedRvec, TcomposedTvec = info[0], info[1]

                        objectPositions = np.array([(0, 0, 0)], dtype=np.float)  # 3D point for projection
                        imgpts, jac = cv2.projectPoints(axis, TcomposedRvec, TcomposedTvec, matrix_coefficients,
                                                        distortion_coefficients)
                        cv2.aruco.drawAxis(image, matrix_coefficients, distortion_coefficients, TcomposedRvec, TcomposedTvec, 0.01)  # Draw Axis
                        relativePoint = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
                        cv2.circle(image, relativePoint, 2, (255, 255, 0))
                        

                    dist_cm = np.linalg.norm(first_marker.tvec - second_marker.tvec)*100
                    if distance_processor is not None:
                        dist_cm = distance_processor(pair, dist_cm)
                    degrees = PositionCalculator().calculate_angle_degrees(first_marker.rvec, second_marker.rvec)
                    if degree_processor is not None:
                        degrees = degree_processor(pair, degrees)
                    print(f"[INFO] Connecting pair: {pair}, distance: {dist_cm:.1f}, degrees: {degrees:.1f}")
                    
                    cX0 = self.Dist[0][0]
                    cY0 = self.Dist[0][1]
                    cX1 = self.Dist[1][0]
                    cY1 = self.Dist[1][1]
                    
                    def get_ray_line(a: Point, b: Point, coeff) -> Line:
                        "returns a and new b with additional length"
                        A=(a.x, a.y )
                        B=(b.x,b.y )
                        
                        lenAB = math.sqrt(math.pow(A[0] - B[0], 2.0) + math.pow(A[1] - B[1], 2.0))
                        b2X = int (B[0] + (B[0] - A[0]) / lenAB * coeff)
                        b2Y = int(B[1] + (B[1] - A[1]) / lenAB * coeff)
                        return Line(a, Point(b2X, b2Y))
                        
                    #cv2.line(image, (cX0, cY0), (cX1, cY1), (255, 0, 255), 2)
                    extended_line = get_ray_line(Point(cX0, cY0), Point(cX1, cY1), 300)
                    #cv2.line(image, (extended_line.x1, extended_line.y1), (extended_line.x2, extended_line.y2), (0, 0, 255), 2)
                    try:
                        pass
                        #markers_midpoint = PositionCalculator().calculate_middle_point(first_marker.rectangle, second_marker.rectangle)
                        #put_text(f"{dist_cm:.1f} cm, {degrees:.1f} deg", Point(markers_midpoint.x, markers_midpoint.y), (0,0,255))
                    except OverflowError:
                        pass
        return image

@dataclass
class TemporaryMarker:
    marker: Marker
    save_time: datetime
class TemporaryStorage:
    def __init__(self) -> None:
        self.storage_time_ms = 1000
        self.storage = [] #marker id: TemporaryMarker
    
    def store_marker(self, marker: Marker):
        self.storage = [tmp_marker for tmp_marker in self.storage if tmp_marker.marker.id != marker.id]
        self.storage.append(TemporaryMarker(marker=marker, save_time=datetime.now()))
        
    def get_markers(self):
        result = []
        for tmpmarker in self.storage:
            time_diff: timedelta = datetime.now() - tmpmarker.save_time 
            diff_milliseconds = time_diff.total_seconds() * 1000
            if diff_milliseconds < self.storage_time_ms:
                result.append(tmpmarker.marker)
            else:
                print(f'[Info] - Will discard marker with id: {tmpmarker.marker.id}, diff: {diff_milliseconds} ms')
        return result
        
distance_measure = DistanceMeasure()
outliers_processor = OutliersProcessor()
temporary_storage = TemporaryStorage()
def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

    def degree_processor(pair, degree):
        return outliers_processor.processed_degree(pair, degree)
    def distance_processor(pair, distance):
        return outliers_processor.processed_distance(pair, distance)
        # If markers are detected
    
    
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            id = ids[i][0]
            if id not in markers_size:
                continue
            marker_size_m = markers_size[id]
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size_m, matrix_coefficients,
                                                                       distortion_coefficients)
            marker = Marker(id=id, tvec=tvec, marker_size_m=marker_size_m, rectangle=None, rvec=rvec,corners=corners[i])
            
            temporary_storage.store_marker(marker)
            # Draw a square around the markers
            #cv2.aruco.drawDetectedMarkers(frame, corners) 
            # Draw Axis
            #cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01) 
        markers = temporary_storage.get_markers()
        marker_storage = MarkerStorage(markers=markers) 
        frame = distance_measure.measure_distance(frame, corners, ids, marker_storage, 
                                                  degree_processor=degree_processor,
                                                  distance_processor=distance_processor,
                                                  matrix_coefficients=matrix_coefficients,
                                                  distortion_coefficients=distortion_coefficients)
    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default='calibration_matrix.npy', help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", default='distortion_coefficients.npy', help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="Type of ArUCo tag to detect")
    ap.add_argument("-i", "--image", help="path to input image containing ArUCo tag (webcam source is not used)")
    args = vars(ap.parse_args())
    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    image = None
    if args["image"]:
        image = cv2.imread(args["image"])
        h,w,_ = image.shape
        width=600
        height = int(width*(h/w))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    video = None
    if image is not None:
        output = pose_esitmation(image, aruco_dict_type, k, d)
        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        sys.exit(0)
    else:
        video = cv2.VideoCapture(0)
        time.sleep(1.0)
    while True:
        
        ret, frame = video.read()
        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)
        
        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if video:
        video.release()
    cv2.destroyAllWindows()