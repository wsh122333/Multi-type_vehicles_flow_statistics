# coding:utf-8
import cv2
from utils.sort import *
from PyQt5.QtCore import  QThread, pyqtSignal
import predict
from config import *

class CounterThread(QThread):
    sin_counterResult = pyqtSignal(np.ndarray)
    sin_runningFlag = pyqtSignal(int)
    sin_videoList = pyqtSignal(list)
    sin_countArea = pyqtSignal(list)
    sin_done = pyqtSignal(int)
    sin_counter_results = pyqtSignal(list)
    sin_pauseFlag = pyqtSignal(int)

    def __init__(self,model,class_names,device):
        super(CounterThread,self).__init__()

        self.model = model
        self.class_names = class_names
        self.device = device

        self.permission = names

        self.colorDict = color_dict

        # create instance of SORT
        self.mot_tracker = Sort(max_age=10, min_hits=2)
        self.countArea = None
        self.running_flag = 0
        self.pause_flag = 0
        self.videoList = []
        self.last_max_id = 0
        self.history = {}  #save history
        #history = {id:{"no_update_count": int, "his": list}}

        self.sin_runningFlag.connect(self.update_flag)
        self.sin_videoList.connect(self.update_videoList)
        self.sin_countArea.connect(self.update_countArea)
        self.sin_pauseFlag.connect(self.update_pauseFlag)

        self.save_dir = "results"
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

    def run(self):
        for video in self.videoList:
            self.last_max_id = 0
            cap = cv2.VideoCapture(video)
            out =  cv2.VideoWriter(os.path.join(self.save_dir,video.split("/")[-1]), cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (1920, 1080))
            frame_count = 0
            while cap.isOpened():
                # print(frame_count)
                if self.running_flag:
                    if not self.pause_flag:
                        ret, frame = cap.read()
                        if ret:
                            if frame_count % 3 == 0:
                                a1 = time.time()
                                frame = self.counter(self.permission, self.colorDict, frame,np.array(self.countArea), self.mot_tracker, video)
                                self.sin_counterResult.emit(frame)

                                out.write(frame)
                                a2 = time.time()
                                print(f"fps: {1 / (a2 - a1):.2f}")
                            frame_count += 1
                        else:
                            break
                    else:
                        time.sleep(0.1)
                else:
                    break

            #restart count for each video
            KalmanBoxTracker.count = 0
            cap.release()
            out.release()

            if not self.running_flag:
                break

        if self.running_flag:
            self.sin_done.emit(1)

    def update_pauseFlag(self,flag):
        self.pause_flag = flag

    def update_flag(self,flag):
        self.running_flag = flag

    def update_videoList(self, videoList):
        print("Update videoList!")
        self.videoList = videoList

    def update_countArea(self,Area):
        print("Update countArea!")
        self.countArea = Area

    def counter(self, permission, colorDict, frame, CountArea, mot_tracker, videoName):

        # painting area
        AreaBound = [min(CountArea[:, 0]), min(CountArea[:, 1]), max(CountArea[:, 0]), max(CountArea[:, 1])]
        painting = np.zeros((AreaBound[3] - AreaBound[1], AreaBound[2] - AreaBound[0]), dtype=np.uint8)
        CountArea_mini = CountArea - AreaBound[0:2]
        cv2.fillConvexPoly(painting, CountArea_mini, (1,))

        objects = predict.yolo_prediction(self.model,self.device,frame,self.class_names)
        objects = filter(lambda x: x[0] in permission, objects)
        objects = filter(lambda x: x[1] > 0.5,objects)
        objects = list(filter(lambda x: pointInCountArea(painting, AreaBound, [int(x[2][0]), int(x[2][1] + x[2][3] / 2)]),objects))

        #filter out repeat bbox
        objects = filiter_out_repeat(objects)

        detections = []
        for item in objects:
            detections.append([int(item[2][0] - item[2][2] / 2),
                               int(item[2][1] - item[2][3] / 2),
                               int(item[2][0] + item[2][2] / 2),
                               int(item[2][1] + item[2][3] / 2),
                               item[1]])
        track_bbs_ids = mot_tracker.update(np.array(detections))

        for i, item in enumerate(objects):
            # x1,y1,x2,y2,id = list(map(lambda x :int(x),item))
            # id_log.add(id)
            # objectName = get_objName(item, objects)

            objectName, province, objectBox = item
            x, y, w, h = objectBox
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

            boxColor = colorDict[objectName]
            cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, thickness=2)
            cv2.putText(frame, objectName, (x1 - 1, y1 - 3), cv2.FONT_HERSHEY_COMPLEX, 0.7, boxColor,
                        thickness=2)



        # painting area
        for i in range(len(CountArea)):
            cv2.line(frame, tuple(CountArea[i]), tuple(CountArea[(i + 1) % (len(CountArea))]), (0, 0, 255), 2)



        if len(track_bbs_ids) > 0:
            for bb in track_bbs_ids:    #add all bbox to history
                id = int(bb[-1])
                objectName = get_objName(bb, objects)
                if id not in self.history.keys():  #add new id
                    self.history[id] = {}
                    self.history[id]["no_update_count"] = 0
                    self.history[id]["his"] = []
                    self.history[id]["his"].append(objectName)
                else:
                    self.history[id]["no_update_count"] = 0
                    self.history[id]["his"].append(objectName)

        counter_results = []
        videoName = videoName.split('/')[-1]
        removed_id_list = []
        for id in self.history.keys():    #extract id after tracking
            self.history[id]["no_update_count"] += 1
            if  self.history[id]["no_update_count"] > 5:
                his = self.history[id]["his"]
                result = {}
                for i in set(his):
                    result[i] = his.count(i)
                res = sorted(result.items(), key=lambda d: d[1], reverse=True)
                objectName = res[0][0]
                counter_results.append([videoName,id,objectName])
                #del id
                removed_id_list.append(id)

        for id in removed_id_list:
            _ = self.history.pop(id)

        if len(counter_results):
            self.sin_counter_results.emit(counter_results)

        print(self.history)


        return frame

    def emit_timeCode(self,time_code):
        self.sin_timeCode.emit(time_code)

def getTwoDimensionListIndex(L,value,pos):
    for i in range(len(L)):
        if L[i][pos] == value:
            return i
    return -1

def filiter_out_repeat(objects):
    objects = sorted(objects,key=lambda x: x[1])
    l = len(objects)
    new_objects = []
    if l > 1:
        for i in range(l-1):
            flag = 0
            for j in range(i+1,l):
                x_i, y_i, w_i, h_i = objects[i][2]
                x_j, y_j, w_j, h_j = objects[j][2]
                box1 = [int(x_i - w_i / 2), int(y_i - h_i / 2), int(x_i + w_i / 2), int(y_i + h_i / 2)]
                box2 = [int(x_j - w_j / 2), int(y_j - h_j / 2), int(x_j + w_j / 2), int(y_j + h_j / 2)]
                if cal_iou(box1,box2) >= 0.7:
                    flag = 1
                    break
            #if no repeat
            if not flag:
                new_objects.append(objects[i])
        #add the last one
        new_objects.append(objects[-1])
    else:
        return objects

    return list(tuple(new_objects))


def cal_iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    i = max(0,(x2-x1))*max(0,(y2-y1))
    u = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) -  i
    iou = float(i)/float(u)
    return iou

def get_objName(item,objects):
    iou_list = []
    for i,object in enumerate(objects):
        x, y, w, h = object[2]
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        iou_list.append(cal_iou(item[:4],[x1,y1,x2,y2]))
    max_index = iou_list.index(max(iou_list))
    return objects[max_index][0]

def pointInCountArea(painting, AreaBound, point):
    h,w = painting.shape[:2]
    point = np.array(point)
    point = point - AreaBound[:2]
    if point[0] < 0 or point[1] < 0 or point[0] >= w or point[1] >= h:
        return 0
    else:
        return painting[point[1],point[0]]





