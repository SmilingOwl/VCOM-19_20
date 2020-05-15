import cv2
import csv
import numpy as np
import imutils

keys = {
    'benign': 1,
    'malignant': 2,
}

def get_descriptors(train_img_path, train_data_path, detector):
    labels = []
    all_descriptors = []
    with open(train_data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            print('Reading image: ' + train_img_path + '/' + row[0] + '.jpg')
            img = cv2.imread(train_img_path + '/' + row[0] + '.jpg', 0) # read image as grayscale
            if img is None:
                continue
            img = imutils.resize(img, width=512)
            keypoints, descriptors = detector.detectAndCompute(img, None) # compute descriptors and keypoints
            if descriptors is None:
                print('none')
                continue
            all_descriptors.extend(descriptors)
    return all_descriptors

def train(bow_extractor, detector, train_data_path, train_img_path):
    print('Training...')
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    img_labels = []
    img_descriptors = []
    with open(train_data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            print('Reading image: ' + train_img_path + '/' + row[0] + '.jpg')
            img = cv2.imread(train_img_path + '/' + row[0] + '.jpg', 0) # read image as grayscale
            if img is None:
                continue
            img = imutils.resize(img, width=512)
            keypoints, descriptors = detector.detectAndCompute(img, None) # compute descriptors and keypoints
            descriptor = bow_extractor.compute(img, keypoints)
            if descriptor is None:
                print('none')
                continue
            img_descriptors.append(descriptor[0])
            img_labels.append(keys[row[1]])
    svm.train(np.array(img_descriptors), cv2.ml.ROW_SAMPLE, np.array(img_labels))
    return svm

def test(bow_extractor, svm, test_img_path, test_data_path, detector):
    print('Testing...')
    img = cv2.imread(test_img_path + '/ISIC_0000000.jpg', 0)
    img = imutils.resize(img, width=512)
    keypoints, descriptors = detector.detectAndCompute(img, None)
    bows = []
    bows.append(bow_extractor.compute(img, keypoints)[0])
    pred = np.squeeze(svm.predict(np.array(bows))[1].astype(int))
    if pred == 1:
        print('benign')
    else
        print('malignant')

def main():
    train_img_path = '../data/training/ISBI2016_ISIC_Part3_Training_Data'
    train_data_path = '../data/training/ISBI2016_ISIC_Part3_Training_GroundTruth.csv'
    test_img_path = '../data/training/ISBI2016_ISIC_Part3_Test_Data'
    detector = cv2.xfeatures2d.SIFT_create()
    all_descriptors = np.array(get_descriptors(train_img_path, train_data_path, detector))
    print('kmeans')
    bow_trainer = cv2.BOWKMeansTrainer(20)
    print('matcher')
    matcher = cv2.FlannBasedMatcher()
    print('bow_extractor')
    bow_extractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
    print('cluster')
    bow_extractor.setVocabulary(bow_trainer.cluster(all_descriptors))
    svm = train(bow_extractor, detector, train_data_path, train_img_path)
    test(bow_extractor, svm, train_img_path, train_data_path, detector)

main()