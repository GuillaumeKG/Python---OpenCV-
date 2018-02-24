import cv2
import logging
import datetime
import os



CL_FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
CL_PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
CL_BODY = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
DEFAULT_CLASSIFIER_LIST = [CL_FRONTAL, CL_PROFILE, CL_BODY]

# Ratio applied to teh image. Must be > 1
DOWNSCALE = 1.1
# Image max size in pixel
MAX_SIZE = 800
# 
MIN_NEIGHBORS = 6

class OpenCVGenericDetection:



    def __init__(self, image_path, archive_folder, debug = False, classifiers = DEFAULT_CLASSIFIER_LIST):
        """
        Constructor
        """
        logging.info("Image: {0}".format(image_path))
        self.image_path = image_path
        self.archive_folder = archive_folder
        self.debug = debug
        self.items = []
        self.items_frames = []

        # Classifier init
        self.set_classifier(classifiers)

        # Define prefix per execution
        self.images_prefix = "{0}_".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))

        # Load image
        self.frame = cv2.imread(image_path)
        logging.info("Image resolution: '{0}x{1}'".format(self.frame.shape[0], self.frame.shape[1]))

        #
        ratio = 1
        if self.frame.shape[1] > MAX_SIZE or self.frame.shape[0] > MAX_SIZE:
            if self.frame.shape[1] > self.frame.shape[0]:
                ratio = float(self.frame.shape[1] / MAX_SIZE)
            else:
                ratio = float(self.frame.shape[0] / MAX_SIZE)

        if ratio != 1:
            newsize = (int(self.frame.shape[1]/ratio), int(self.frame.shape[0]/ratio))
            logging.info("Image resizing: {0}".format(newsize))
            self.frame = cv2.resize(self.frame, newsize)

        if self.debug:
            cv2.imshow("preview", self.frame)
            cv2.waitKey()
            cv2.destroyAllWindows()


    def process_image(self):
        """
        """
        self.find_items()
        if len(self.items) > 0:
            self.extract_items_frames()
            self.archive_with_items()
            self.archive_items_frames()
            self.archive_with_items()

    def set_classifier(self, list):
        """
        TO be overloaded
        """
        self.classifiers = list

    def find_items(self):
        """
        Find items in a frame
        """
        logging.info("Searching items in image")
        all_items = []

        for classifier in self.classifiers:
            items = classifier.detectMultiScale(self.frame, 
                scaleFactor = DOWNSCALE, 
                minNeighbors = MIN_NEIGHBORS)
            all_items.extend(items)    
            logging.info("{0}: {1} items".format(classifier.__class__, len(items)))
        
        logging.info("Items count: {0}".format(len(all_items)))
        
        logging.info("Items = {0}".format(all_items))
        self.items = all_items


    def extract_items_frames(self):
        """
        Extract items frames from original frame
        """
        logging.info("Extraction of frame's items in progress... {0} items to process".format(len(self.items)))

        items_frames = []
        logging.info("self.items: {0}".format(self.items))
        for f in self.items:
            logging.info("f: {0}".format(f))
            # We extract coordinates of th frame
            x, y, w, h = f
            item_frame = self.frame[y:y+h, x:x+w]
            items_frames.append({
                "frame": item_frame,
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

        if self.debug:
            cv2.imshow("item", item_frame)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        self.items_frames = items_frames

    def get_items_frames(self, grayscale = False):
        """
        Return items' frames and their coordinates in a list
        """
        if not grayscale:
            return self.items_frames
        
        items_frames = []
        for item_frame in self.items_frames:
            item_frame["frame"] = cv2.cvtColor(item_frame["frame"], cv2.COLOR_BGR2GRAY)
            items_frames.append(item_frame)

        return items_frames

    def add_label(self, text, x, y):
        """
        Add label on original frame
        """
        if y >11:
            y = y - 5
        
        cv2.putText(self.frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 0), 2)

    def archive_items_frames(self):
        """
        Archive frame of each item as an image
        """
        logging.info("Archiving items...")
        idx = 0

        for item_frame in self.items_frames:
            a_frame = item_frame["frame"]
            image_name = "{0}_item_{1}.jpg".format(self.images_prefix, idx)
            cv2.imwrite(os.path.join(self.archive_folder, image_name), a_frame)
            idx += 1

    def archive_with_items(self):
        """
        Archive original frame of with square for each item
        """
        logging.info("Archiving original image with items' frames")
        for f in self.items:
            x, y, h, w = f
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        archive_full_name = "{0}_full.jpg".format(self.images_prefix)
        logging.info("Original file saved in {0}".format(os.path.join(self.archive_folder, archive_full_name)))
        cv2.imwrite(os.path.join(self.archive_folder, archive_full_name), self.frame)

        if self.debug:
            cv2.imshow("item", self.frame)
            cv2.waitKey()
            cv2.destroyAllWindows()


