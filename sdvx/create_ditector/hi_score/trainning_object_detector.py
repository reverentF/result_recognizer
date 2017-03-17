import os
import sys
import dlib

img_dir = "."

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = False
options.C = 5
options.num_threads = 2
options.be_verbose = True

training_xml_path = os.path.join(img_dir, "training.xml")
testing_xml_path = os.path.join(img_dir, "test.xml")

dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

print("")
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))