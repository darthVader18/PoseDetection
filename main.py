from detection import *

detect = Detector()

# detect.onImage("input/76.jpg")

video_input = "input/train.mp4"
vi = "rtsp://192.168.25.164:8080/h264_ulaw.sdp"

detect.onVideo(video_input)
