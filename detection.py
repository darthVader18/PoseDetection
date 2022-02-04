from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np


class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        Pose = None
        image = cv2.imread(imagePath)
        results = {}
        predictions = self.predictor(image)

        viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                         instance_mode=ColorMode.IMAGE)

        output = viz.draw_instance_predictions(
            predictions["instances"].to("cpu"))
        output1 = predictions["instances"].to("cpu")
        keypoints = output1.pred_keypoints.detach().numpy()
        results['keypoints'] = keypoints.tolist()
        # print(results)

        def calculate_angle(a, b, c):
            a = np.array(a)  # First
            b = np.array(b)  # Mid
            c = np.array(c)  # End

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle > 180.0:
                angle = 360-angle
            return angle

        try:
            nose = keypoints[0][0][0], keypoints[0][0][1]
            left_hip = keypoints[0][11][0], keypoints[0][11][1]
            left_knee = keypoints[0][13][0], keypoints[0][13][1]

            angle = calculate_angle(nose, left_hip, left_knee)
            print(angle)

            right_hip = keypoints[0][12][0], keypoints[0][12][1]
            right_ankle = keypoints[0][16][0], keypoints[0][16][1]
            left_ankle = keypoints[0][15][0], keypoints[0][15][1]
            right_knee = keypoints[0][14][0], keypoints[0][14][1]

            angle1 = calculate_angle(right_ankle, left_knee, left_ankle)
            print(f"angle1: {angle1}")

            angle2 = calculate_angle(nose, right_hip, right_knee)
            print(f"angle2: {angle2}")

            right_leg_angle = calculate_angle(
                right_hip, right_knee, right_ankle)
            print(f"right_leg_angle: {right_leg_angle}")

            left_leg_angle = calculate_angle(
                left_hip, left_knee, left_ankle)
            print(f"left_leg_angle: {left_leg_angle}")

            if angle > 165 and 0 < angle1 < 10 and (right_leg_angle > 175 or left_leg_angle > 175):
                Pose = "standing"
                print(Pose)
            elif angle > 165 and 10 < angle1 < 40 and (right_leg_angle < 180 or left_leg_angle < 180):
                Pose = "walking"
                print(Pose)
            elif angle > 115 and angle1 > 40:
                Pose = "running"
                print(Pose)
            elif angle < 115 and angle2 < 115:
                Pose = "sitting"
                print(Pose)
        except:
            pass

        x = cv2.rectangle(output.get_image(), (0, 0),
                          (150, 50), (245, 117, 16), -1)
        cv2.putText(x, Pose,
                    (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("result", x)
        # cv2.imshow("Result", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    def onVideo(self, videoPath):
        Pose = None
        results = {}
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened() == False):
            print("Error opening the file....")

        (success, image) = cap.read()

        while success:
            predictions = self.predictor(image)

            # viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            #                  instance_mode=ColorMode.IMAGE)

            # output = viz.draw_instance_predictions(
            #     predictions["instances"].to("cpu"))
            # output1 = predictions["instances"].to("cpu")
            # print(output1)
            # keypoints = output1.pred_keypoints.detach().numpy()
            # print(predictions["instances"].pred_boxes.to('cpu'))
            # output = v.draw_instance_predictions(
            #     predictions["instances"].to("cpu"))

            v = Visualizer(
                image[:, :, ::-1],
                metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                scale=0.8,
            )

            def calculate_angle(a, b, c):
                a = np.array(a)  # First
                b = np.array(b)  # Mid
                c = np.array(c)  # End

                radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                    np.arctan2(a[1]-b[1], a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi)

                if angle > 180.0:
                    angle = 360-angle
                return angle
            pose = []
            for itm in predictions["instances"].pred_keypoints.to('cpu'):
                try:
                    nose = itm[0, 0].tolist(), itm[0, 1].tolist()
                    left_hip = itm[11, 0].tolist(), itm[11, 1].tolist()
                    left_knee = itm[13, 0].tolist(), itm[13, 1].tolist()
                    right_hip = itm[12, 0].tolist(), itm[12, 1].tolist()
                    right_ankle = itm[16, 0].tolist(), itm[16, 1].tolist()
                    left_ankle = itm[15, 0].tolist(), itm[15, 1].tolist()
                    right_knee = itm[14, 0].tolist(), itm[14, 1].tolist()

                    angle = calculate_angle(nose, left_hip, left_knee)
                    angle1 = calculate_angle(
                        right_ankle, left_knee, left_ankle)
                    angle2 = calculate_angle(nose, right_hip, right_knee)
                    right_leg_angle = calculate_angle(
                        right_hip, right_knee, right_ankle)
                    left_leg_angle = calculate_angle(
                        left_hip, left_knee, left_ankle)

                    if angle > 165 and 0 < angle1 < 10 and (right_leg_angle > 175 or left_leg_angle > 175):
                        Pose = "standing"
                        # v.draw_text(Pose, (box[0], box[1]))
                        pose.append(Pose)
                        # print(Pose)
                    elif angle > 165 and 10 < angle1 < 40 and (right_leg_angle < 180 or left_leg_angle < 180):
                        Pose = "walking"
                        # v.draw_text(Pose, (box[0], box[1]))
                        pose.append(Pose)
                        # print(Pose)
                    elif angle > 115 and angle1 > 40:
                        Pose = "running"
                        # v.draw_text(Pose, (box[0], box[1]))
                        pose.append(Pose)
                        # print(Pose)
                    elif angle < 115 and angle2 < 115:
                        Pose = "sitting"
                        # v.draw_text(Pose, (box[0], box[1]))
                        pose.append(Pose)
                        # print(Pose)
                    # v.draw_text(Pose, (box[0], box[1]))
                except:
                    pass
            # for sc in predictions["instances"].scores.to('cpu'):
            print(pose)
            # v.draw_instance_predictions(pose)
            for c in pose:
                for box, label in zip(predictions["instances"].pred_boxes.to('cpu'), pose):
                    v.draw_box(box)
                    v.draw_text(label, (box[0], box[1]))

            # print(box[0].tolist(), box[1].tolist())

            v = v.get_output()
            img = v.get_image()[:, :, ::-1]

            resize = cv2.resize(img, (560, 640))
            cv2.imshow('result', resize)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()
