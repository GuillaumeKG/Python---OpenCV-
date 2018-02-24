from GenericDetection import OpenCVGenericDetection
from GenericRecognition import OpenCVGenericRecognition
import logging
import os

logging.basicConfig(filename='test_log.log',level=logging.DEBUG,\
      format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

TRAINSET_PATH = "f:\\temp\\trainset"
ARCHIVE_PATH = "f:\\temp\\archives"
NEW_IMG_DIR =  "f:\\temp\\img_to_process"

# Load and prepare Recognizer
recognizer = OpenCVGenericRecognition(TRAINSET_PATH)
recognizer.load_trainset()
recognizer.train()


# Detect face on new dataset
for filename in os.listdir(NEW_IMG_DIR):
    detector = OpenCVGenericDetection(
        image_path = os.path.join(NEW_IMG_DIR, filename), 
        archive_folder = ARCHIVE_PATH, 
        debug=False)

    detector.find_items()
    detector.extract_items_frames()

    for item in detector.get_items_frames(grayscale = True):
        found, identity, confidence = recognizer.recognize(item["frame"]) 
        if found:
            label = "{0} {1}".format(identity, confidence)
            x = item["x"]
            y = item["y"]
            detector.add_label(label, x, y)
    #detector.archive_items_frames()
    detector.archive_with_items()

