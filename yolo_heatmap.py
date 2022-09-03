import cv2
import numpy as np
from imutils.video import VideoStream

# video source: https://www.mediafire.com/file/ccp10qr81kuzwzk/video.mp4/file

video_file = "video.mp4"
weights_file = "yolov4-tiny.weights"

cfg_file = "yolov4-tiny.cfg"
classname_file = "coco.names"
detected_class = "person"


frame_width = 1280
frame_height = 720
conf_threshold = 0.5
nms_threshold = 0.4

cell_size = 40
n_cols = frame_width // cell_size
n_rows = frame_height // cell_size
alpha = 0.4

# helper function
def load_model(weights=weights_file,cfg=cfg_file):
    print("Load YOLOv4 model: Done")
    net = cv2.dnn.readNet(weights,cfg)

    return net

def get_classes(file_name=classname_file):
    classes = None
    with open(file_name, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def get_output_layer(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def get_row_col(x, y):
    row = y// cell_size
    col = x //cell_size
    return row, col

def detect_objects(frame, output_layers, net):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs

def display(img, class_id, x, y, x_plus_w, y_plus_h):

    global heat_matrix
    r, c = get_row_col( (x_plus_w + x)//2, (y_plus_h + y)//2)
    heat_matrix[r,c] +=1


    label = str(classes[class_id])
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_grid(image, n_rows, n_cols):
    for i in range(n_rows):
        start_point = (0, (i + 1) * cell_size)
        end_point = (frame_width, (i + 1) * cell_size)
        color = (255, 255, 255)
        thickness = 1
        image = cv2.line(image, start_point, end_point, color, thickness)

    for i in range(n_cols):
        start_point = ((i + 1) * cell_size, 0)
        end_point = ((i + 1) * cell_size, frame_height)
        color = (255, 255, 255)
        thickness = 1
        image = cv2.line(image, start_point, end_point, color, thickness)

    return image

################ main ###########################
heat_matrix = np.zeros((n_rows, n_cols))

video = VideoStream(video_file).start()

classes = get_classes(classname_file)

# generate color list
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = load_model(weights_file,cfg_file)

codec = cv2.VideoWriter_fourcc(*"MJPG")

result = cv2.VideoWriter('result.avi' , codec, 10, (frame_width, frame_height))
while True:
    frame = video.read()

    output_layers = get_output_layer(net)
    outs = detect_objects(frame,output_layers,net)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if (confidence > conf_threshold) and (classes[class_id]==detected_class):

                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)

                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)

                x = center_x - w / 2
                y = center_y - h / 2

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        display(frame, class_ids[i], round(x), round(y), round(x + w), round(y + h))
    
    from skimage.transform import resize
    temp_heat_matrix = heat_matrix.copy()
    temp_heat_matrix = resize(temp_heat_matrix, (frame_height, frame_width))
    temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
    temp_heat_matrix = np.uint8(temp_heat_matrix*255)

    image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)

    cv2.addWeighted(image_heat, alpha, frame, (1-alpha)*2, 0, frame)

    
    result.write(frame)
    cv2.imshow("Frame",frame)

    if cv2.waitKey(1)== ord('q'):
        break

result.release()
cv2.destroyAllWindows()