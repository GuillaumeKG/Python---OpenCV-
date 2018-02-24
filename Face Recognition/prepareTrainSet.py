from GenericDetection import OpenCVGenericDetection
import logging
import os

logging.basicConfig(filename='test_log.log',level=logging.DEBUG,\
      format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

NEW_IMG_PATH = "E:\\_\\Exportation sans titre"
TRAINSET_PATH = "f:\\temp\\trainset"

# Detect face on new dataset
for filename in os.listdir(NEW_IMG_PATH):
    detector = OpenCVGenericDetection(
        image_path = os.path.join(NEW_IMG_PATH, filename), 
        archive_folder = TRAINSET_PATH, 
        debug=False)

    detector.find_items()
    detector.extract_items_frames()
    detector.archive_items_frames()
    #detector.archive_with_items()