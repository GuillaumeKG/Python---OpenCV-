import cv2
import numpy
import logging
from enum import Enum
import os
import sys

class OpenCVGenericRecognition:

    ALGO_LBPH = 1
    ALGO_FISHER = 2
    ALGO_EIGEN = 3

    THRESHOLD_CONFIDENCE = 70

    def __init__(self, trainset_path, reco_algo = ALGO_LBPH):
        """
        """
        logging.info("Trainset: {0}".format(trainset_path))
        self.trainset_path = trainset_path
        self.reco_algo = reco_algo

        self.resize_faces = (170, 170)

        self.model = None
        self.trainset_images = []
        self.trainset_index = []
        self.trainset_identities = []

    def load_trainset(self):
        """
        """
        logging.info("Trainset loading...")

        c = 0

        for dirname, dirnames, filenames in os.walk(self.trainset_path):
            for subdirname in dirnames:
                self.trainset_identities.append(subdirname)
                logging.info("- identity '{0}'...".format(subdirname))
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    try:
                        im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                        im = cv2.resize(im, self.resize_faces)

                        self.trainset_images.append(numpy.asarray(im, dtype=numpy.uint8))
                        self.trainset_index.append(c)

                    except IOError (errno, strerror):
                        logging.error("I/O error({0}): {1}".format(errno, strerror))

                    except:
                        logging.error("Unexepected error:", sys.exc_info()[0])
                        raise
                c += 1


    def train(self):
        """
        """
        logging.info("Trainset Processing...")
        
        if self.reco_algo == self.ALGO_EIGEN:
            self.model = cv2.face.EigenFaceRecognizer_create()
        elif self.reco_algo == self.ALGO_FISHER:
            self.model = cv2.face.FisherFaceRecognizer_create()
        else:
            self.model = cv2.face.LBPHFaceRecognizer_create()
        
        self.model.train(numpy.asarray(self.trainset_images), numpy.asarray(self.trainset_index))

    def recognize(self, frame):
        """
        """
        frame = cv2.resize(frame, self.resize_faces)
        [idx, confidence] = self.model.predict(frame)
        found_identity = self.trainset_identities[idx]

        if confidence < self.THRESHOLD_CONFIDENCE:
            identity = found_identity
            found = True
        else:
            identity = "N/A ({0}".format(found_identity)
            found = False
        
        return found, identity, int(confidence)
        