from flask import Flask, render_template, jsonify, request, Response
import cv2              
import time
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import datetime
from utils.functions import *
from copy import deepcopy

import requests
import base64
import json  
import threading

from utils.service.TFLiteFaceAlignment import * 
from utils.service.TFLiteFaceDetector import * 
from utils.functions import *


app = Flask(__name__)

path = "/home/xavier01/facerec_cctv_fe/"
# path = "./"

fa = CoordinateAlignmentModel(path + "utils/service/weights/coor_2d106.tflite")


parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default= path + 'weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25',
                    help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true",
                    default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.7,
                    type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4,
                    type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true",
                    default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.8, type=float,
                    help='visualization_threshold')
args = parser.parse_args()

extend_pixel = 50
predict_labels = []


url = 'https://dohubapps.com/user/langiocn/5000/facerecog'

# save_path = 'detected_faces/'

# if not os.path.exists(save_path):
#     os.makedirs(save_path)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def small_face_detection(image):

    img = np.float32(image)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # Forward pass
    loc, conf, landms = net(img) 

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(
        0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(
        0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    return np.concatenate((dets, landms), axis=1).astype(int)

def face_recognize(frame, face_I):
    _, encimg = cv2.imencode(".jpg", frame)
    img_byte = encimg.tobytes()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    new_img_str = "data:image/jpeg;base64," + img_str


    _, encimg_2 = cv2.imencode(".jpg", face_I)
    img_byte_2 = encimg_2.tobytes()
    img_str_2 = base64.b64encode(img_byte_2).decode('utf-8')
    new_img_str_2 = "data:image/jpeg;base64," + img_str_2
    # Specify the file path where you want to create or write to a file
    # file_path = f"image_{time.time()}.txt"

    # # Open the file for writing (use 'w' for write mode)
    # with open(file_path, 'w') as file:
    # # Write the string to the file
    #     file.write(new_img_str)


    heads={'Content-Type':'application/json', 'x-api-key': '862d14253a4ffcad4c63a9f1d3c07449b4339ec3b33684259bfaec2358b10968'}
    payload = {'img': img_str}
    response = requests.post(url, json=payload, headers=heads, timeout=1000)

    try:
        # Check if the response contains JSON content
        try:
            json_string = response.content  
            json_data = json.loads(json_string)
            # print(response)
            # Access specific elements within the JSON data
            # For example, if the JSON contains a 'results' key
            if 'names' in json_data and len(json_data['names']) > 0:
                identity = json_data['names'][0]
                if identity !=  "Unknown":
                    print("Identity:", identity)

                    # cur_url = url + '/faces/' + identity
                    # cur_profile_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                    predict_labels.append(['id', identity, new_img_str_2, "data:image/jpeg;base64," + json_data['imgs'][0], time.time()])

                    # if identity != 'Unknown':
                    #     save_image(save_path + str(time.time()) + '_' + resI, frame)
                    # else:
                    #     save_image(save_path + str(time.time()) + '_' + resI + '.jpg', frame)

        except ValueError as e:
            print("Response is not in JSON format:", e)

    except requests.exceptions.RequestException:
        print(response.text)

def save_image(save_path_, frame_):
    # Save the frame as an image
    cv2.imwrite(save_path_, frame_)

def frame_extract(path):
  vidObj = cv2.VideoCapture(path) 
  success = 1
  while success:
      success, image = vidObj.read()
      if success:
          yield image

def face_detect():

    # Initialize the webcam
    cap = cv2.VideoCapture(0) 

    # save_path = 'frames/'

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    prev_frame_time = 0
    new_frame_time = 0
    queue = []
    extend_pixel = 50

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the frame height
    total_fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second

    print('width:', width, 'px')
    print('height:', height, 'px')
    print('total_fps:', total_fps, 'fps')


    for idx,frame in enumerate(frame_extract(0)):
        if(idx % 1 == 0):

            final_img = deepcopy(frame)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
        
            temp_boxes  = small_face_detection(frame)

            draw_box(final_img, temp_boxes, color=(125, 255, 125))

            # Find landmarks of each face
            temp_marks = fa.get_landmarks(frame, temp_boxes)

            # print('Detect face in:', str(new_frame_time_0 - prev_frame_time), ' ms')
            for bbox_I, landmark_I in zip(temp_boxes, temp_marks):
                # draw_landmark(final_img, landmark_I, color=(125, 255, 125))
            
                # Show rotated raw face image
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])
                xmin -= extend_pixel
                xmax += extend_pixel
                ymin -= 2 * extend_pixel
                ymax += extend_pixel

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                xmax = frame_width if xmax >= frame_width else xmax
                ymax = frame_height if ymax >= frame_height else ymax

                face_I = frame[ymin:ymax, xmin:xmax]
                face_I = align_face(face_I, landmark_I[34], landmark_I[88])

                # cv2.imshow('Rotated face image', face_I)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # image_filename = f"image_{time.time()}.jpg"
                # save_image(save_path + image_filename, face_I)

                if(idx % 5 == 0):
                    queue = [t for t in queue if t.is_alive()]
                    if len(queue) < 3:
                        # queue.append(threading.Thread(target=face_recognize, args=(face_I,)))
                        queue.append(threading.Thread(target=face_recognize, args=(frame, face_I,)))
                        queue[-1].start()

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))

            cv2.putText(final_img, '{0}/{1} fps'.format(fps, total_fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

            # cv2.imshow('Webcam', final_img)

            # # Exit the loop if the user presses the 'q' key
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


            # Convert the frame to a jpeg image
            ret, jpeg = cv2.imencode('.jpg', final_img)

            # Return the image as bytes
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

def non_face_detect():

    # Initialize the webcam

    cam_add = 'rtsp://admin:pilot2214@192.168.50.14:554/Streaming/channels/1/'
    cap = cv2.VideoCapture(cam_add) 

    # save_path = 'frames/'

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    prev_frame_time = 0
    new_frame_time = 0
    queue = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the frame height
    total_fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second

    print('width:', width, 'px')
    print('height:', height, 'px')
    print('total_fps:', total_fps, 'fps')
    index_frame = 0
    for idx,frame in enumerate(frame_extract(cam_add)):
        index_frame += 1
        if(index_frame % 5 == 0):
            queue = [t for t in queue if t.is_alive()]
            if len(queue) < 3:
                queue.append(threading.Thread(target=face_recognize, args=(frame, frame,)))
                queue[-1].start()


            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))

            cv2.putText(frame, '{0}/{1} fps'.format(fps, total_fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

            # cv2.imshow('Webcam', final_img)

            # # Exit the loop if the user presses the 'q' key
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            index_frame = 0

            # Convert the frame to a jpeg image
            ret, jpeg = cv2.imencode('.jpg', frame)

            # Return the image as bytes
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()



@app.route('/video_feed_0')
def video_feed_0():
    return Response(face_detect(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/non_face_detect_0')
def non_face_detect_0():
    return Response(non_face_detect(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return '<h1>Server running</h1>'

@app.route('/data')
def data():
    global predict_labels
    if len(predict_labels) > 3:
        predict_labels = predict_labels[-3:]
    newest_data = list(reversed(predict_labels))
    return jsonify({'info': newest_data})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    # elif args.network == "resnet50":
    #     cfg = cfg_re50
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    cudnn.benchmark = True
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    net = net.to(device)

    raw_img = ""
    final_img = ""
    last_time = datetime.datetime.now().timestamp()

    # face_detect()
    app.run(host='0.0.0.0', debug=True, port=6060)

