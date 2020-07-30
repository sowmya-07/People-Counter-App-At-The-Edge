"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    client =  mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client
def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    prob_threshold = args.prob_threshold
    current_count = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            current_count = current_count + 1
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    return frame, current_count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    single_input_mode = False
    
    current_req_id = 0
    last_count = 0
    total_count = 0
    start_time = 0

    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
   
    
    infer_network.load_model(model = args.model, device = args.device,
                             cpu_extension = args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    prob_threshold = args.prob_threshold
    single_image_mode = False

    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    else:
        input_stream = args.input
        assert os.path.isfile(args.input),"input file doesnt exist"

    
    

    ### TODO: Loop until stream is over ###
    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)
    if not cap.isOpened():
        log.error("video source cant open")
        out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('M','J','P','G'),                              30,(100,100))
    
    prob_threshold = args.prob_threshold
    width = int(cap.get(3))
    height = int(cap.get(4))
    count = 0
    ### TODO: Read from the video capture ###
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        #preprocess_image = person_detection(input_image)
        
        preprocess_image = cv2.resize(frame,(net_input_shape[3],net_input_shape[2]))
        preprocess_image = preprocess_image.transpose((2,0,1))
        preprocess_image = preprocess_image.reshape(1,*preprocess_image.shape)

        ### TODO: Start asynchronous inference for specified request ###
        inference_time = time.time()
        
        infer_network.exec_net(preprocess_image)


        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            duration_time = time.time()-inference_time
            #fps = cap.get(cv2.CAP_PROP_FPS)
            #frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            #duration_time = float(frame_count)/float(fps)
            result = infer_network.get_output()
            frame, current_count = draw_boxes(frame, result, args, width, height)
            
            inf_time_message = "Inference time: {:.3f}ms".format(duration_time*1000)
            cv2.putText(frame, inf_time_message, (15,15),cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255,0,0),1)

                


            ### TODO: Get the results of the inference request ###
            

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            if current_count > last_count and last_count == 0:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total":total_count}))
            if current_count < last_count and current_count == 0 and int(time.time() -
                                                                         start_time) >=1:
                duration = int(time.time()-start_time)
                client.publish("person/duration", json.dumps({"duration":duration}))
                client.publish("person",json.dumps({'count':current_count}))
            if current_count > 1:
                client.publish("person", json.dumps({"count":1}))
            else:
                client.publish("person", json.dumps({"count":current_count}))
            last_count = current_count 
            ### Topic "person/duration": key of "duration" ###
            ### TODO: Send the frame to the FFMPEG server ###
            if key_pressed == 27:
                break
            ### TODO: Write an output image if `single_image_mode` ###
            if single_input_mode:
                frame=cv2.resize(frame,(1980,1080))
            cv2.imwrite('output_image.jpg', frame)
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
    
    cap.release()
    client.disconnect()
    cv2.destroyAllWindows()              
            
            


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
