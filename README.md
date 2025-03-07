import cv2
import imutils
import numpy as np
import argparse

def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.1)

    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {person}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    cv2.putText(frame, 'Status: Detecting', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons: {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

    if person == 1:
        cv2.putText(frame, 'No persons detected', (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('output', frame)
    return frame

def detectByPathVideo(path, writer):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        print('Error: Video not found or cannot be opened.')
        return

    print('Detecting people...')
    while True:
        check, frame = video.read()
        if not check:
            break

        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        frame = detect(frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def detectByCamera(writer):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print('Error: Camera could not be opened.')
        return

    print('Detecting people...')
    while True:
        check, frame = video.read()
        if not check:
            break

        frame = detect(frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    if image is None:
        print('Error: Image not found.')
        return

    image = imutils.resize(image, width=min(800, image.shape[1]))
    result_image = detect(image)

    if output_path:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(1)
    cv2.destroyAllWindows()

def humanDetector(args):
    image_path = args["image"]
    video_path = args["video"]
    camera = args["camera"]

    writer = None
    if args["output"] and not image_path:
        writer = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))

    if camera:
        print("[INFO] Opening Web Cam.")
        detectByCamera(writer)
    elif video_path:
        print("[INFO] Opening Video from path.")
        detectByPathVideo(video_path, writer)
    elif image_path:
        print("[INFO] Opening Image from path.")
        detectByPathImage(image_path, args["output"])

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="Path to Video File")
    arg_parse.add_argument("-i", "--image", default=None, help="Path to Image File")
    arg_parse.add_argument("-c", "--camera", action="store_true", help="Use camera")
    arg_parse.add_argument("-o", "--output", type=str, default=None, help="Path to output video file")
    
    # Return dictionary of arguments
    return vars(arg_parse.parse_args([]))  # Simulated empty args for testing

if _name_ == "_main_":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor().getDefaultPeopleDetector())

    args = argsParser()
    humanDetector(args)
