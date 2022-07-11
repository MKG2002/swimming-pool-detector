import cv2
import numpy as np
import os

net = cv2.dnn.readNet("model.weights", "testing.cfg")

classes = ["Pool"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    print("Enter the path of Input Image or enter  STOP  to end the programme")
    while True:
        img_path = input()
        if img_path == "STOP" :
            print("IT WAS A NICE JOURNEY OF UDYAM IIT (BHU) VARANASI 2022")
            break
        if not os.path.exists(img_path):
            print("Please enter Correct Path or enter  STOP  to end the programme")
            continue
        break
    if img_path == "STOP" :
        break
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1.6, fy=1.6)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            print(w*h)
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0 , 0 , 255), 3)
            cv2.putText(img, label, (x, y), font, 1, (0 , 0 , 255), 2)
    cv2.imshow(' ' , img)
    cv2.waitKey(0)

