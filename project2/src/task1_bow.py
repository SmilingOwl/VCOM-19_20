import cv2
import csv
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description="Feature detection")
parser.add_argument('-d', '--descriptors', default=False, dest='read_desc', action='store_true')
parser.add_argument('-b', '--bow', default=False, dest='read_bow', action='store_true')
parser.add_argument('-t', '--train', default=False, dest='read_svm', action='store_true')
args = parser.parse_args()

keys = {
    'benign': 0,
    'malignant': 1,
}

train_img_path = '../data/task1/training/ISBI2016_ISIC_Part3_Training_Data'
train_data_path = '../data/task1/training/ISBI2016_ISIC_Part3_Training_GroundTruth.csv'
test_img_path = '../data/task1/test/ISBI2016_ISIC_Part3_Test_Data'
test_data_path = '../data/task1/test/ISBI2016_ISIC_Part3_Test_GroundTruth.csv'

# Get descriptors using sift
def get_descriptors(detector):
    if args.read_desc:
        print('Loading descriptors...')
        with open('../data/feature_detection/all_descriptors.pkl', 'rb') as inputfile:
            return pickle.load(inputfile)

    print('Calculating descriptors...')
    labels = []
    all_descriptors = []
    with open(train_data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            img = cv2.imread(train_img_path + '/' + row[0] + '.jpg', 0) # read image as grayscale
            keypoints, descriptors = detector.detectAndCompute(img, None) # compute descriptors and keypoints
            all_descriptors.extend(descriptors)
    with open('../data/feature_detection/all_descriptors.pkl', 'wb') as outputfile:
        pickle.dump(all_descriptors, outputfile)
    return all_descriptors

# Train SVM classifier
def train(bow_extractor, detector):
    if args.read_svm:
        print('Loading trained svm...')
        return cv2.ml.SVM_load('../data/feature_detection/svm.pkl')
    print('Training...')
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    img_labels = []
    img_descriptors = []
    with open(train_data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            img = cv2.imread(train_img_path + '/' + row[0] + '.jpg', 0) # read image as grayscale
            keypoints, descriptors = detector.detectAndCompute(img, None) # compute descriptors and keypoints
            descriptor = bow_extractor.compute(img, keypoints)
            img_descriptors.append(descriptor[0])
            img_labels.append(keys[row[1]])
    svm.train(np.array(img_descriptors), cv2.ml.ROW_SAMPLE, np.array(img_labels))
    svm.save('../data/feature_detection/svm.pkl')
    return svm

# Use SVM classifier on test dataset
def test(bow_extractor, svm, detector):
    print('Testing...')
    total = 0
    right = 0
    benign_right = 0
    benign_wrong = 0 # malignant picture classified as benign
    malignant_right = 0
    malignant_wrong = 0 # benign picture classified as malignant
    with open(test_data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            img = cv2.imread(test_img_path + '/' + row[0] + '.jpg', 0) # read image as grayscale
            keypoints, descriptors = detector.detectAndCompute(img, None) # compute descriptors and keypoints
            b = []
            b.append(bow_extractor.compute(img, keypoints)[0])
            prediction = svm.predict(np.array(b))
            pred = np.squeeze(prediction[1].astype(int))
            if pred == int(float(row[1])):
                right += 1
                if pred == 0:
                    benign_right += 1
                else:
                    malignant_right += 1
            else:
                if pred == 0:
                    benign_wrong += 1
                else:
                    malignant_wrong += 1
            total += 1
    print('Total: ' + str(total))
    print('Malignant correctly identified: ' + str(malignant_right))
    print('Benign correctly identified: ' + str(benign_right))
    print('Malignant identified as benign: ' + str(benign_wrong))
    print('Benign identified as malignant: ' + str(malignant_wrong))
    print('Accuracy: ' + str(float(right) / float(total)))

def main():
    # create feature detector with SIFT
    detector = cv2.xfeatures2d.SIFT_create()
    # obtain descriptors
    all_descriptors = np.array(get_descriptors(detector))
    # create or load bow
    if args.read_bow:
        print('Loading bag of words...')
        with open('../data/feature_detection/bow.pkl', 'rb') as inputfile:
            bow_cluster = pickle.load(inputfile)
    else:
        print('Training Bag of Words...')
        bow_trainer = cv2.BOWKMeansTrainer(300)
        bow_cluster = bow_trainer.cluster(all_descriptors)
        with open('../data/feature_detection/bow.pkl', 'wb') as outputfile:
            pickle.dump(bow_cluster, outputfile)
    matcher = cv2.FlannBasedMatcher()
    bow_extractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
    bow_extractor.setVocabulary(bow_cluster)
    # obtain svm classifier
    svm = train(bow_extractor, detector)
    # use classifier on test dataset
    test(bow_extractor, svm, detector)

main()