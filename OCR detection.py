import torch
import re
import cv2
import numpy as np
import easyocr
import pytesseract
import os
####--------------------------------- OCR libraries-------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
EASY_OCR = easyocr.Reader(['en'])

OCR_TH = 0.2
####----------------------------- Image Detection---------------------------------------
def image_detection (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    model.classes = [5,8,10]
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

#### ---------------------------- function to extract text from documents --------------------------------------


def recognize_text(img, coords,reader,region_threshold):

    xmin, ymin, xmax, ymax = coords

    nplate = img[ymin:ymax,xmin:xmax] ### cropping bounding Boxes
    cv2.imwrite('static/new.png',nplate)



    ocr_results = pytesseract.image_to_string(nplate)
    print(ocr_results)


### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plotting_boxes(results, frame):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    labels, cord = results
    n = len(labels)
    print(n)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates



            coords = [x1,y1,x2,y2]
            print(coords)

            plate_num = recognize_text(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)

    return frame



### ---------------------------------------------- loading model -----------------------------------------------------

# def main(image_path = None)
print(f' loading model......')
# model = torch.hub.load('./yolov5-master')
model = torch.hub.load('.', 'custom',source = 'local', path='static/best.pt',force_reload=True)


classes = [model.names[5],model.names[8],model.names[10]]
print(classes)

### --------------- for detection on image --------------------

print('working with image')
frame = cv2.imread(r'data\images_documents\test\images\34-90503_0009_jpg.rf.abf348da33514aa4c62e9e67144a7ff1.jpg') ### reading the image
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
results = image_detection(frame, model = model) ### DETECTION HAPPENING HERE
frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
frame = plotting_boxes(results, frame)
# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# cv2.imwrite('static/gray.png',gray)
# cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result
#
# while True:
#     # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#
#     cv2.imshow("img_only", frame)
#
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         print(f"[INFO] Exiting. . . ")
#
#         cv2.imwrite(f"{img_out_name}",frame) ## if you want to save he output result.
#
#         break







