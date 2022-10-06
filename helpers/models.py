from dataclasses import dataclass
from typing import Optional

class Point:    
    # pass arr=[0,1] or x=0,=1
    def __init__(self, *args, **kwargs) -> None:
        try:
            arr = args[0]
            if len(args) == 2:
                self.x = args[0]
                self.y = args[1]
            else:
                self.x = arr[0]
                self.y = arr[1]
        except Exception as exc:
            print(exc)
            self.x = kwargs['x']
            self.y = kwargs['y']   
class Line:    
    # pass arr=[x1,x2,y1,y2] or x1=0,x2=1,y1=1,y2=2
    def __init__(self, *args, **kwargs) -> None:
        try:
            arr = args[0]
            if len(args) == 2:
                if isinstance(args[0], Point) and isinstance(args[1], Point):
                    self.x1 = args[0].x
                    self.x2 = args[1].x
                    self.y1 = args[0].y
                    self.y2 = args[1].y
            elif len(args) == 4:
                self.x1 = args[0]
                self.x2 = args[1]
                self.y1 = args[2]
                self.y2 = args[3]
            else:
                self.x1 = arr[0]
                self.x2 = arr[1]
                self.y1 = arr[2]
                self.y2 = arr[3]
        except Exception as exc:
            print(exc)
            self.x1 = kwargs['x1']
            self.x2 = kwargs['x2']
            self.y1 = kwargs['y1']
            self.y2 = kwargs['y2']   
            
            
@dataclass
class Rectangle:
    topLeft: Point
    topRight: Point
    bottomRight: Point
    bottomLeft: Point


@dataclass  
class Marker:
    id: int
    tvec: any
    marker_size_m: float
    rectangle: Optional[Rectangle]
    rvec: any
    corners: any