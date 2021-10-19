import cv2
from queue import Queue
from yolov5_detect_image import Y5Detect, draw_boxes_tracking, draw_det_when_track, draw_single_pose
import time
from kthread import KThread
import numpy as np
from mot_sort.mot_sort_tracker import Sort
from mot_sort import untils_track
from collections import deque

import torch
from PoseEstimate.PoseEstimateLoader import SPPE_FastPose
from Actionsrecognition.ActionsEstLoader import TSSTG

y5_model = Y5Detect(weights="model_head/yolov5s.pt")
class_names = y5_model.class_names
mot_tracker = Sort(class_names)

inp_pose = (224, 160)
pose_model = SPPE_FastPose("resnet50", inp_pose[0], inp_pose[1], device="cuda")

# Actions Estimate.
action_model = TSSTG()


class InfoCam(object):
    def __init__(self, cam_name):
        self.cap = cv2.VideoCapture(cam_name)


def video_capture(cam, frame_detect_queue):
    frame_count = 0

    cam.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    while cam.cap.isOpened():
        ret, frame_ori = cam.cap.read()
        # time.sleep(0.01)
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2RGB)
        frame_detect_queue.put([image_rgb, frame_count])
        print("frame_count: ", frame_count)
        frame_count += 1

    cam.cap.release()


def inference(cam, frame_detect_queue, detections_queue): #, tracking_queue):
    while cam.cap.isOpened():
        image_rgb, frame_count = frame_detect_queue.get()
        boxes, labels, scores, detections_sort = y5_model.predict_sort(image_rgb, label_select=["person"])
        # for i in range(len(scores)):
        #     detections_tracking = bboxes[i].append(scores[i])
        detections_queue.put([boxes, labels, scores, image_rgb, detections_sort, frame_count])
        # tracking_queue.put([detections_tracking])

    cam.cap.release()


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def to_tlwh(tlbr):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    ret = tlbr.copy()
    # ret[2:] += ret[:2]
    box = []
    for bbox in ret:
        w = int(bbox[2]) - int(bbox[0])
        h = int(bbox[3]) - int(bbox[1])
        box.append([int(bbox[0]) + w/2, int(bbox[1]) + h/2, w, h])
    return box


def tracking(cam, detections_queue, pose_queue):
    """
    :param cam:
    :param pose_queue:
    :param detections_queue:
    :param draw_queue:
    :return:
    Tracking using SORT. Hungary + Kalman Filter.
    Using mot_tracker.update()
    Input: detections [[x1,y1,x2,y2,score,label],[x1,y1,x2,y2,score, label],...], use np.empty((0, 5)) for frames without detections
    Output: [[x1,y1,x2,y2,id1, label],[x1,y1,x2,y2,id2, label],...]
    """
    region_track = np.array([[0, 0],
                             [2560, 0],
                             [2560, 1440],
                             [0, 1440]])

    while cam.cap.isOpened():
        boxes, labels, scores, image_rgb, detections_sort, frame_count = detections_queue.get()
        if len(boxes) == 0:
            detections = np.empty((0, 5))
        else:
            detections = detections_sort
            # check and select the detection is inside region tracking
            # detections = untils_track.select_bbox_inside_polygon(detections, region_track)

        track_bbs_ids, unm_trk_ext = mot_tracker.update(detections, image=image_rgb)
        # print("labels, scores", labels, scores)
        # print(track_bbs_ids)
        if len(track_bbs_ids) > 0:
            a = 0
        pose_queue.put([track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count])

    cam.cap.release()


def pose_estimate(cam, pose_queue, pose_draw_queue, draw_queue, action_data_queue):
    data_action_rec = []
    pre_track_id = []

    while cam.cap.isOpened():
        track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count = pose_queue.get()
        if len(track_bbs_ids) > 0:
            boxes_detect_pose = torch.as_tensor(track_bbs_ids[:, 0:4])
            scores = torch.as_tensor(np.ones(len(track_bbs_ids)))
            poses = pose_model.predict(image_rgb, boxes_detect_pose, scores)
            key_points_pose = [np.concatenate((ps['keypoints'].numpy(), ps['kp_score'].numpy()), axis=1) for ps in poses]
            # print("key_points_pose: ", key_points_pose)

            current_track_id = track_bbs_ids[:, -1]
            if len(data_action_rec) > 0:
                pre_track_id = list(map(lambda d: d['track_id'], data_action_rec))

            for i in range(len(current_track_id)):
                key_points = key_points_pose[i]
                if current_track_id[i] not in pre_track_id:
                    # Create new track
                    key_points_list = deque(maxlen=30)
                    key_points_list.append(key_points)
                    data_action_rec.append({
                        "track_id": current_track_id[i],
                        "key_points": key_points_list
                    })
                else:
                    idx_pre_track = pre_track_id.index(current_track_id[i])
                    data_action_rec[idx_pre_track]["key_points"].append(key_points)

                    # Update key points for track

            pose_draw_queue.put(key_points_pose)
            draw_queue.put([track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count])
            action_data_queue.put([data_action_rec, image_rgb])
    cam.cap.release()


def action_recognition(cam, action_data_queue):
    while cam.cap.isOpened():
        """pts (30, 13, 3)
        pts:  [[[194.06215    119.31413      0.9018644 ]
              [223.90459    134.1215       0.88993466]
              [228.14021    129.88216      0.710703  ]
              ...
              [247.1983     234.21288      0.6688115 ]
              [240.91792    274.54507      0.81022084]
              [242.95183    261.7644       0.6869712 ]]
              ...
              ...
              [[200.65388    109.802765     0.92303926]
              [181.53549    133.70076      0.65679896]
              [176.54015    140.087        0.63335484]
              ...
              [156.45757    245.17003      0.87906563]
              [152.14519    277.01837      0.3391189 ]
              [143.70879    297.46173      0.86148417]]]
        """
        data_action_rec, image_rgb = action_data_queue.get()
        # print("data_action_rec", len(data_action_rec))
        # print("data_action_rec", len(data_action_rec[0]["key_points"]))
        for i in range(len(data_action_rec)):
            if len(data_action_rec[0]["key_points"]) == 30:
                out = action_model.predict(data_action_rec[0]["key_points"], image_rgb.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                print(action_name)
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

    cam.cap.release()


def drawing(cam, tracking_queue, frame_final_queue, pose_draw_queue, show_det=False):
    while cam.cap.isOpened():
        track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count = tracking_queue.get()
        key_points_pose = pose_draw_queue.get()
        frame_origin = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if frame_origin is not None:
            image = draw_boxes_tracking(frame_origin, track_bbs_ids, scores=scores, labels=labels,
                                        class_names=class_names, track_bbs_ext=unm_trk_ext)

            for i in range(len(key_points_pose)):
                draw_single_pose(frame_origin, key_points_pose[i], joint_format='coco')

            if show_det:
                image = draw_det_when_track(frame_origin, boxes, scores=scores, labels=labels,
                                            class_names=class_names)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if frame_final_queue.full() is False:
                frame_final_queue.put([image, frame_count])
            else:
                time.sleep(0.001)
    cam.cap.release()


def save_debug_image(frame_count, image):
    cv2.imwrite("/home/vuong/Desktop/Project/GG_Project/green-clover-montessori/new_core/debug_image/test_" + str(
        frame_count) + ".png", image)


def main():
    frame_detect_queue = Queue(maxsize=1)
    pose_queue = Queue(maxsize=1)
    action_data_queue = Queue(maxsize=1)
    pose_draw_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    draw_queue = Queue(maxsize=1)
    frame_final_queue = Queue(maxsize=1)
    input_path = "https://minio.core.greenlabs.ai/clover/fall_detection/fall_detect.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20211018%2F%2Fs3%2Faws4_request&X-Amz-Date=20211018T063245Z&X-Amz-Expires=432000&X-Amz-SignedHeaders=host&X-Amz-Signature=e711d8737a4f22831169fdfc7008e66e22eb024a853b430ca47c0ff0b19d9809"
    cam = InfoCam(input_path)

    thread1 = KThread(target=video_capture, args=(cam, frame_detect_queue))
    thread2 = KThread(target=inference, args=(cam, frame_detect_queue, detections_queue))
    thread3 = KThread(target=tracking, args=(cam, detections_queue, pose_queue))
    thread4 = KThread(target=pose_estimate, args=(cam, pose_queue, pose_draw_queue, draw_queue, action_data_queue))
    thread5 = KThread(target=action_recognition, args=(cam, action_data_queue))
    thread6 = KThread(target=drawing, args=(cam, draw_queue, frame_final_queue, pose_draw_queue))

    thread_manager = []
    thread1.daemon = True  # sẽ chặn chương trình chính thoát khi thread còn sống.
    thread1.start()
    thread_manager.append(thread1)
    thread2.daemon = True
    thread2.start()
    thread_manager.append(thread2)
    thread3.daemon = True
    thread3.start()
    thread_manager.append(thread3)
    thread4.daemon = True
    thread4.start()
    thread_manager.append(thread4)
    thread5.daemon = True
    thread5.start()
    thread_manager.append(thread5)
    thread6.daemon = True
    thread6.start()
    thread_manager.append(thread6)

    while cam.cap.isOpened():
        cv2.namedWindow('output')
        image, frame_count = frame_final_queue.get()
        image = cv2.resize(image, (1400, 640))
        cv2.imshow('output', image)
        # if frame_count >= 1550 and (frame_count <= 1800):
        #     KThread(target=save_debug_image, args=(frame_count, image)).start()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow('output')
            break

    for t in thread_manager:
        if t.is_alive():
            t.terminate()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

