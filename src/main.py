import os
from typing import Counter
import cv2
import csv
import logging as log
import time
from argparse import ArgumentParser
import numpy as np
import pyautogui
import tkinter
from PIL import Image, ImageTk
from input_feeder import InputFeeder
from face_detection import FaceDetector
from head_pose_estimation import HeadPoseEstimator
from facial_landmarks_detection import FacialLandmarksDetector
from gaze_estimation import GazeEstimator
from threading import Thread
import winsound
import pandas as pd
from data_evaluation import data_preprocess, return_coordinates
WIDTH, HEIGHT = pyautogui.size()
print(WIDTH,HEIGHT)
filename ='dot_test.csv'
def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    '''
    parser.add_argument("-mfd", "--model_face_detection", required=True, type=str,
                        #help="Path to an xml file with a trained face detection model.")
    parser.add_argument("-mhpe", "--model_head_pose_estimation", required=True, type=str,
                        help="Path to an xml file with a trained head pose estimation model.")
    parser.add_argument("-mfld", "--model_facial_landmarks_detection", required=True, type=str,
                        help="Path to an xml file with a trained facial landmarks detection model.")
    parser.add_argument("-mge", "--model_gaze_estimation", required=True, type=str,
                        help="Path to an xml file with a trained gaze estimation model.")
    '''
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Specify 'video', 'image' or 'cam' (to work with camera).")
    parser.add_argument("-i", "--input_path", required=False, type=str, default=None,
                        help="Path to image or video file.")
    parser.add_argument("-o", "--output_path", required=False, type=str, default="results",
                        help="Path to image or video file.")                        
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
    parser.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    parser.add_argument("--show_input", help="Optional. Show input video",
                      default=False, action="store_true")
    parser.add_argument("--move_mouse", help="Optional. Move mouse based on gaze estimation",
                      default=False, action="store_true")
    parser.add_argument("--calibrate", help="Optional. Calibrate the screen",
                      default=False, action="store_true")
    parser.add_argument("--show_video", help="Optional. Show video output",
                      default=False, action="store_true")
    parser.add_argument("--record", help="Record gaze angles",
                      default=False, action="store_true")
    return parser

def infer_on_stream(args):
    try:
        log.basicConfig(
            level=log.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                log.FileHandler("app.log"),
                log.StreamHandler()
            ])
            

        start_model_load_time=time.time()

        face_detector = FaceDetector('models\\intel\\face-detection-adas-binary-0001\\FP32-INT1\\face-detection-adas-binary-0001')
        facial_landmarks_detector = FacialLandmarksDetector('models\\intel\\landmarks-regression-retail-0009\\FP32\\landmarks-regression-retail-0009')
        head_pose_estimator = HeadPoseEstimator('models\\intel\\head-pose-estimation-adas-0001\\FP32\\head-pose-estimation-adas-0001')
        gaze_estimator = GazeEstimator('models\\intel\\gaze-estimation-adas-0002\\FP32\\gaze-estimation-adas-0002')
        face_detector.load_model()
        facial_landmarks_detector.load_model()
        head_pose_estimator.load_model()
        gaze_estimator.load_model()
        np_avgs = data_preprocess()
        total_model_load_time = time.time() - start_model_load_time
        log.info("Model load time: {:.1f}ms".format(1000 * total_model_load_time))

        output_directory = os.path.join(args.output_path + '\\' + args.device)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        feed = InputFeeder(args.input_type, args.input_path)
        feed.load_data()
        out_video = feed.get_out_video(output_directory)

        frame_counter = 0
        start_inference_time=time.time()
        total_prepocess_time = 0

        # Write Gaze angles to file
        # 
        #if args.record:
        file_b = open(filename, 'a+', newline='')
        writer = csv.writer(file_b)
        writer.writerow(['eye_0_x' ,'eye_0_y', 'eye_1_x', 'eye_1_y' ,'gaze_angle_x', 'gaze_angle_y'])

        if args.calibrate:
            file_a = open(filename, 'a+', newline='')
            writer = csv.writer(file_a)
            writer.writerow(['Point', 'Frame', 'Time', 'Gaze_angle_x', 'Gaze_angle_y'])
            #Now open calibration bit
            pilImage = Image.open("bin/calibration.png")
            
            showPIL(pilImage)
            thread = Thread(target = showPIL,args=(pilImage,))
            thread.start()
        start_time = time.time()
        # Do OpenCV stuff
        file_name = "images/tesla.mp4"
        window_name = "window"
        interframe_wait_ms = 30
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out = cv2.VideoWriter('tesla_out.avi',fourcc, 20.0, (480,320))
        cap2 = cv2.VideoCapture(file_name)
        radius = 10
        color = (255, 0, 0)
        if not cap2.isOpened():
            print("Error: Could not open video.")
            exit()
        counter = 0
        while True:
            
            
            
            while(cap2.isOpened()):
                ret2, frame2 = cap2.read()
                
                if cv2.waitKey(interframe_wait_ms) & 0x7F == ord('q'):
                    print("Exit requested.")
                    break
                current_time = time.time() - start_time
                
                try:
                    frame = next(feed.next_batch())
                except StopIteration:
                    break
                frame_counter += 1

                face_boxes = face_detector.predict(frame)
                
                for face_box in face_boxes:
                    
                    face_image = get_crop_image(frame, face_box)
                    eye_boxes, eye_centres = facial_landmarks_detector.predict(face_image)
                    left_eye_image, right_eye_image = [get_crop_image(face_image, eye_box) for eye_box in eye_boxes]
                    head_pose_angles = head_pose_estimator.predict(face_image)
                    gaze_x, gaze_y = gaze_estimator.predict(right_eye_image, head_pose_angles, left_eye_image)
                    # Get x,y coords for gaze
                    
                    x_coord_decimal, y_coord_decimal = return_coordinates(np_avgs,gaze_x,gaze_y)
                    x_coord = int(x_coord_decimal*1366)
                    y_coord = int(y_coord_decimal*766)
                    centre_coordinates = (x_coord,y_coord)
                    #print("Gaze Angles:",gaze_x,gaze_y )
                    #print("Centre coordinates:",centre_coordinates)
                    
                    # Get eye x and y locations from eye centres:
                    eye_0_x = eye_centres[0][0]
                    eye_0_y = eye_centres[0][1]
                    eye_1_x = eye_centres[1][0]
                    eye_1_y = eye_centres[1][1]

                    if args.show_video & ret2==True:
                        image2 = cv2.circle(frame, centre_coordinates, radius, color, thickness=3)
                        # write the frame
                        out.write(image2)

                        cv2.imshow(window_name, image2)
                    
                    if args.calibrate:
                        writer.writerow(["1", frame_counter, current_time, gaze_x, gaze_y])
                    if args.record:
                        writer.writerow([eye_0_x ,eye_0_y, eye_1_x, eye_1_y,gaze_x, gaze_y])
                    if args.show_input:
                        cv2.imshow('im', frame)
                    break
                
            
                if out_video is not None:
                    out_video.write(frame)
                if args.input_type == "image":
                    cv2.imwrite(os.path.join(output_directory, 'output_image.jpg'), frame)

                key_pressed = cv2.waitKey(60)
                if key_pressed == 27:
                    break

            # This is for knowing how far along the processing
            # of a video fil is 
            #print(counter)
            counter+= 1
            total_time=time.time()-start_inference_time
            total_inference_time=round(total_time, 1)
            fps=frame_counter/total_inference_time
            log.info("Inference time:{:.1f}ms".format(1000* total_inference_time))
            log.info("Input/output preprocess time:{:.1f}ms".format(1000* total_prepocess_time))
            log.info("FPS:{}".format(fps))
            if args.record:
                df1= pd.read_csv("Participants\P_13\P_13_1491923431229_3_-study-dot_test.webm_gazePredictionsDone.csv")
                df2= pd.read_csv(filename)
                df3 = pd.concat([df2,df1], axis=1)
                df3.to_csv('Participants/P_13/merged_dot_test.csv', index = False)
                print(df2)
                # Make a beep so I know processing is done
            frequency = 2500  # Set Frequency To 2500 Hertz
            duration = 1000  # Set Duration To 1000 ms == 1 second
            winsound.Beep(frequency, duration)
            with open(os.path.join(output_directory, 'stats.txt'), 'w') as f:
                f.write(str(total_inference_time)+'\n')
                f.write(str(total_prepocess_time)+'\n')
                f.write(str(fps)+'\n')
                f.write(str(total_model_load_time)+'\n')
                if args.calibrate:
                    file_a.close()
                if args.record:
                    file_b.close()
                
            feed.close()
            
            cv2.destroyAllWindows()

            break
    except Exception as e:
        log.exception("Something wrong when running inference:" + str(e))

def get_crop_image(image, box):
    xmin, ymin, xmax, ymax = box
    crop_image = image[ymin:ymax, xmin:xmax]
    return crop_image

def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)

# THis is where Calibration Image is shown
def showPIL(pilImage):
        root = tkinter.Tk()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.overrideredirect(1)
        root.geometry("%dx%d+0+0" % (w, h))
        root.focus_set()    
        root.bind("<Escape>", lambda e: (e.widget.withdraw(), e.widget.quit()))
        canvas = tkinter.Canvas(root,width=w,height=h)
        canvas.pack()
        canvas.configure(background='black')
        imgWidth, imgHeight = pilImage.size
        if imgWidth > w or imgHeight > h:
            ratio = min(w/imgWidth, h/imgHeight)
            imgWidth = int(imgWidth*ratio)
            imgHeight = int(imgHeight*ratio)
            pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(pilImage)
        imagesprite = canvas.create_image(w/2,h/2,image=image)
        root.mainloop()

if __name__ == '__main__':
    main()