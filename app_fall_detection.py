import cv2
from queue import Queue
from yolov5_detect_image import Y5Detect, draw_boxes_tracking, draw_det_when_track, draw_single_pose, draw_data_action
import time
from kthread import KThread
import numpy as np
from mot_sort.mot_sort_tracker import Sort
from mot_sort import untils_track
from collections import deque

import torch
from PoseEstimate.PoseEstimateLoader import SPPE_FastPose
from Actionsrecognition.ActionsEstLoader import TSSTG

from flask import Response
from flask import Flask, request
from flask import render_template
import threading

outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

input_path = None


y5_model = Y5Detect(weights="model_yolov5/yolov5s.pt")
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
        # print("frame_count: ", frame_count)
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


def pose_estimate(cam, pose_queue, pose_draw_queue, action_data_queue):
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
            # Delete pre_track_id not in current_track_id, track is deleted.
            track_id_delete = np.setdiff1d(pre_track_id, current_track_id)
            if len(track_id_delete) > 0:
                for track_id in track_id_delete:
                    index_del = pre_track_id.index(track_id)
                    del data_action_rec[index_del]
                    pre_track_id.remove(track_id)
                    a = 0

            if len(key_points_pose) == len(current_track_id):
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
            else:
                print("len(key_points_pose) != len(current_track_id)")

            pose_draw_queue.put(key_points_pose)
            # draw_queue.put([track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count])
            action_data_queue.put([data_action_rec, image_rgb, track_bbs_ids, boxes, labels, scores, unm_trk_ext, frame_count])
    cam.cap.release()


def action_recognition(cam, action_data_queue, draw_queue):
    action_name = 'pending..'
    while cam.cap.isOpened():
        data_action_rec, image_rgb, track_bbs_ids, boxes, labels, scores, unm_trk_ext, frame_count = action_data_queue.get()
        data_action = data_action_rec.copy()
        # print("data_action_rec", len(data_action_rec))
        # print("data_action_rec", len(data_action_rec[0]["key_points"]))
        print("len(data_action_rec)", len(data_action))
        for i in range(len(data_action)):
            if len(data_action[i]["key_points"]) == 30:
                """pts (30, 13, 3)"""
                pts = np.array(data_action[i]["key_points"], dtype=np.float32)
                out = action_model.predict(pts, image_rgb.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                print(action_name)
                # action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                action_name_data = {'action_name': action_name}
                data_action[i].update(action_name_data)
                a = 0
        draw_queue.put([track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count, data_action])
    cam.cap.release()


def drawing(cam, tracking_queue, frame_final_queue, pose_draw_queue, show_det=False):
    while cam.cap.isOpened():
        track_bbs_ids, boxes, labels, scores, unm_trk_ext, image_rgb, frame_count, data_action = tracking_queue.get()
        key_points_pose = pose_draw_queue.get()
        frame_origin = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if frame_origin is not None:
            image = draw_data_action(frame_origin, track_bbs_ids, track_bbs_ext=unm_trk_ext, data_action=data_action)

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

    global outputFrame, lock, input_path
    cv2_show = False
    if cv2_show:
        input_path = "https://minio.core.greenlabs.ai/clover/fall_detection/fall_detection1.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20211021%2F%2Fs3%2Faws4_request&X-Amz-Date=20211021T073136Z&X-Amz-Expires=432000&X-Amz-SignedHeaders=host&X-Amz-Signature=26e19be6f47f0b10da7ada4562d628376e4123235d4a430be70e7ceed2ca1567"
    cam = InfoCam(input_path)

    thread1 = KThread(target=video_capture, args=(cam, frame_detect_queue))
    thread2 = KThread(target=inference, args=(cam, frame_detect_queue, detections_queue))
    thread3 = KThread(target=tracking, args=(cam, detections_queue, pose_queue))
    thread4 = KThread(target=pose_estimate, args=(cam, pose_queue, pose_draw_queue, action_data_queue))
    thread5 = KThread(target=action_recognition, args=(cam, action_data_queue, draw_queue))
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
        image, frame_count = frame_final_queue.get()
        image = cv2.resize(image, (1400, 640))
        with lock:
            outputFrame = image.copy()
        # if frame_count >= 1550 and (frame_count <= 1800):
        #     KThread(target=save_debug_image, args=(frame_count, image)).start()
        if cv2_show:
            cv2.imshow('output', image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow('output')
                break

    for t in thread_manager:
        if t.is_alive():
            t.terminate()
    cv2.destroyAllWindows()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/fall_detection")
def stream():
    # return the rendered template
    return render_template("index.html")


@app.route("/start_video", methods=['POST', 'GET'])
def start_video():
    global input_path
    #  https://flask.palletsprojects.com/en/2.0.x/quickstart/#a-minimal-application
    if request.method == 'POST':
        file = request.form["name"]
        input_path = file
        t = KThread(target=main)
        t.daemon = True
        t.start()
    # return the rendered template
    return "<p>OK!</p>"


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # main()
    # # # start the flask app
    app.run(host="0.0.0.0", port="44444", debug=True,
            threaded=True, use_reloader=False)


