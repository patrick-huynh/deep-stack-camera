import cv2
# Import DeepStack's Python SDK
from deepstack_sdk import ServerConfig, Detection


# Function to draw detections and object names on camera frames
def draw_detections(img, detections):
    for detection in detections:
        output_font_scale = 0.8e-3 * img.shape[0]
        label = detection.label
        img = cv2.rectangle(
                    img,
                    (detection.x_min, detection.y_min),
                    (detection.x_max, detection.y_max),
                    (0,146,224),
                    2
                )
        img = cv2.putText(
                        img=img,
                        text=label + " ( " + str(100*detection.confidence)+"% )",
                        org=(detection.x_min-10, detection.y_min-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=output_font_scale,
                        color=(0,146,224),
                        thickness=2
                    )
        
    return img




if __name__=="__main__":

    # Initiate Connection to DeepStack
    config = ServerConfig("http://localhost:80")
    detection = Detection(config)

    # Initiate video capure to webcam
    capture = cv2.VideoCapture(1)

    while(True):
        # Capture the video frame
        ret, frame = capture.read()
    
        if ret:
            # Detect the Frame with DeepStack using the Python SDK
            detections = detection.detectObject(frame,output=None)
            print(detections)
            # Draw the detections on the frame
            frame = draw_detections(frame, detections)

            # Display the frame and the detections
            cv2.imshow('frame', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
