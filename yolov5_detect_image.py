
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
import cv2
from glob import glob
from unidecode import unidecode


class Y5Detect:
    def __init__(self, weights):
        """
        :param weights: 'yolov5s.pt'
        """
        self.weights = weights
        self.model_image_size = 640
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.model, self.device = self.load_model(use_cuda=True)

        stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(self.model_image_size, s=stride)
        self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def load_model(self, use_cuda=False):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_cuda else 'cpu')

        model = attempt_load(self.weights, map_location=device)
        return model, device

    def preprocess_image(self, image_rgb):
        # Padded resize
        img = letterbox(image_rgb.copy(), new_shape=self.image_size)[0]

        # Convert
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, image_rgb, show=False):
        image_rgb_shape = image_rgb.shape
        img = self.preprocess_image(image_rgb)
        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred,
                                   self.conf_threshold,
                                   self.iou_threshold,)
        bboxes = []
        labels = []
        scores = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_rgb_shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    with torch.no_grad():
                        x1 = xyxy[0].cpu().data.numpy()
                        y1 = xyxy[1].cpu().data.numpy()
                        x2 = xyxy[2].cpu().data.numpy()
                        y2 = xyxy[3].cpu().data.numpy()
                        #                        print('[INFO] bbox: ', x1, y1, x2, y2)
                        bboxes.append(list(map(int, [x1, y1, x2, y2])))
                        label = self.class_names[int(cls)]
                        #                        print('[INFO] label: ', label)
                        labels.append(label)
                        score = conf.cpu().data.numpy()
                        #                        print('[INFO] score: ', score)
                        scores.append(float(score))
        if show:
            if bboxes is not None:
                image_show = draw_boxes(image_bgr, bboxes, scores=scores, labels=labels, class_names=self.class_names)
                cv2.namedWindow('detections', cv2.WINDOW_NORMAL)
                cv2.imshow('detections', image_show)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return bboxes, labels, scores

    def predict_sort(self, image_rgb, label_select=list):
        image_rgb_shape = image_rgb.shape
        img = self.preprocess_image(image_rgb)
        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred,
                                   self.conf_threshold,
                                   self.iou_threshold,)
        bboxes = []
        labels = []
        scores = []
        preds = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_rgb_shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    with torch.no_grad():
                        x1 = xyxy[0].cpu().data.numpy()
                        y1 = xyxy[1].cpu().data.numpy()
                        x2 = xyxy[2].cpu().data.numpy()
                        y2 = xyxy[3].cpu().data.numpy()
                        label = self.class_names[int(cls)]
                        if label in label_select:
                            bboxes.append(list(map(int, [x1, y1, x2, y2])))
                            labels.append(label)
                            score = conf.cpu().data.numpy()
                            scores.append(float(score))
                            preds.append(np.array([int(x1), int(y1), int(x2), int(y2), float(score), int(cls)]))

        return bboxes, labels, scores, np.array(preds)


def draw_boxes(image, boxes, scores=None, labels=None, class_names=None, line_thickness=2, font_scale=1.0,
               font_thickness=2):
    num_classes = len(class_names)
    if scores is not None and labels is not None:
        for b, l, s in zip(boxes, labels, scores):
            if class_names is None:
                class_name = 'person'
                class_id = 0
            elif l not in class_names:
                class_id = int(l)
                class_name = class_names[class_id]
            else:
                class_name = l
                class_id = class_names.index(l)

            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)
            color = (45, 255, 255)
            label = '-'.join([class_name, score])

            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                        font_thickness)
    else:
        color = (0, 255, 0)
        for b in boxes:
            xmin, ymin, xmax, ymax = list(map(int, b))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
    return image


def draw_det_when_track(image, boxes, scores=None, labels=None, class_names=None, line_thickness=2, font_scale=1.0,
               font_thickness=2):
    num_classes = len(class_names)
    if scores is not None and labels is not None:
        for b, l, s in zip(boxes, labels, scores):
            if class_names is None:
                class_name = 'person'
                class_id = 0
            elif l not in class_names:
                class_id = int(l)
                class_name = class_names[class_id]
            else:
                class_name = l
                class_id = class_names.index(l)

            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)
            color = (255, 255, 255)
            label = '-'.join([class_name, score])
            cv2.rectangle(image, (xmin-3, ymin-3), (xmax+3, ymax+3), color, line_thickness)
    else:
        color = (0, 255, 0)
        for b in boxes:
            xmin, ymin, xmax, ymax = list(map(int, b))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
    return image


def draw_boxes_tracking(image, track_bbs_ids, scores, labels, class_names, track_bbs_ext=None, line_thickness=1, font_scale=1.0,
                        font_thickness=2):
    for b in track_bbs_ids:
        xmin, ymin, xmax, ymax, track_id = list(map(int, b))
        color = (0, 0, 255)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)
        # put id track to image
        cv2.putText(image, str(track_id), (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),
                    3)
    if track_bbs_ext is not None:
        for b in track_bbs_ext:
            xmin, ymin, xmax, ymax = list(map(int, b))
            color = (0, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)
            cv2.putText(image, "track_ext", (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        1)

    return image


def draw_data_action(image, track_bbs_ids, track_bbs_ext=None, data_action=None, line_thickness=1):
    for idx, b in enumerate(track_bbs_ids):
        xmin, ymin, xmax, ymax, track_id = list(map(int, b))
        color = (0, 0, 255)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)
        # put id track to image
        cv2.putText(image, str(track_id), (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),
                    3)
        if len(track_bbs_ids) == len(data_action):
            if data_action is not None and "action_name" in data_action[idx]:
                if data_action[idx]["action_name"] == "Fall Down":
                    cv2.putText(image, unidecode("Té Ngã"), (xmin, ymin + 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),
                                2)
                if data_action[idx]["action_name"] == "Lying Down":
                    cv2.putText(image, unidecode("Nằm"), (xmin, ymin + 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),
                                2)
                if data_action[idx]["action_name"] == "Walking" or data_action[idx]["action_name"] == "Standing":
                    cv2.putText(image, unidecode("Đứng or Đi"), (xmin, ymin + 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),
                                2)
    if track_bbs_ext is not None:
        for b in track_bbs_ext:
            xmin, ymin, xmax, ymax = list(map(int, b))
            color = (0, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)
            cv2.putText(image, "track_ext", (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        1)

    return image


def draw_single_pose(frame, pts, joint_format='coco'):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    PURPLE = (255, 0, 255)

    """COCO_PAIR = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head
                 (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                 (17, 11), (17, 12),  # Body
                 (11, 13), (12, 14), (13, 15), (14, 16)]"""
    COCO_PAIR = [(0, 13), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (13, 7), (13, 8),  # Body
                 (7, 9), (8, 10), (9, 11), (10, 12)]
    POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                    # Nose, LEye, REye, LEar, REar
                    (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                    # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                    (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    LINE_COLORS = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222),
                   (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255),
                   (255, 156, 127), (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    MPII_PAIR = [(8, 9), (11, 12), (11, 10), (2, 1), (1, 0), (13, 14), (14, 15), (3, 4), (4, 5),
                 (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)]

    if joint_format == 'coco':
        l_pair = COCO_PAIR
        p_color = POINT_COLORS
        line_color = LINE_COLORS
    elif joint_format == 'mpii':
        l_pair = MPII_PAIR
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED,BLUE,BLUE]
    else:
        NotImplementedError

    part_line = {}
    pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
    for n in range(pts.shape[0]):
        if pts[n, 2] <= 0.05:
            continue
        cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
        part_line[n] = (cor_x, cor_y)
        cv2.circle(frame, (cor_x, cor_y), 3, p_color[n], -1)

    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(frame, start_xy, end_xy, line_color[i], int(1*(pts[start_p, 2] + pts[end_p, 2]) + 1))
    return frame


if __name__ == '__main__':
    y5_model = Y5Detect(weights="/home/thien/Desktop/Huy/Fall_detection/model_yolov5/yolov5s.pt")

    img_path = "/home/thien/Desktop/Huy/Fall_detection/Image/"

    list_image_test = glob(img_path + "*.jpg")
    for image_test in list_image_test:
        image_bgr = cv2.imread(image_test)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        y5_model.predict(image_rgb, show=True)