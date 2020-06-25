import argparse
import datetime
import platform
from Classifier import Classifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(
        description="Compute trajectory features from OpenPose points to Human Action Recognition"
    )

    parser.add_argument("--test_name", type=str, required=True,
                        help="Experiment test name.")

    parser.add_argument("--features_file_filter", nargs='+', type=str,
                        default='*.*',
                        help="Filter for features files")

    #parser.add_argument("--base_path", nargs='+', type=str, required=True,
    #                    help="Base features path.")

    parser.add_argument("--base_path", type=str, required=True,
                        help="Base features path.")

    parser.add_argument("--base_path2", type=str,
                        help="Base features path.")

    parser.add_argument("--label_path", type=str,
                        help="Labels path.")

    parser.add_argument("--number_cluster", type=int,
                        default=30,
                        help="Number of cluster for to build BoG using FV.")

    parser.add_argument("--c_parameter", type=int,
                        default=1,
                        help="C parameter for SVM classifier.")

    parser.add_argument("--use_train_test_val", type=int,
                        default=1,
                        help="True if dataset is divides into train/test/validation.")

    args = parser.parse_args()

    print(args)
    print(args.base_path)
    print(args.features_file_filter[0])
    #perform_loocv_fusion(args)
    #perform_loocv_classifiers(args)
    perform_loocv_fusion_new(args)
    #if len(args.base_path) > 1:
    #    perform_loocv_fusion(args)
    #else:
    #    perform_loocv_classifiers(args)


def perform_loocv_fusion_new(args):
    # Instantiate Classifier
    list = []
    while len(list) < 100:
        random_number = np.random.randint(999)
        if random_number not in list:
            list.append(random_number)

    print(list)
    ctd = 0
    for i in list:
        ctd = ctd + 1
        print('STEP ', i)
        #random_number = np.random.randint(33)
        random_number = i
        print(random_number)
        test_name = args.test_name + '_' + str(ctd)
        print(test_name)
        classifier = Classifier(classifier=SVC(kernel='linear'),
                                no_clusters=args.number_cluster,
                                gmm_random_state=random_number,
                                no_samples=None)

        classifier.OsName = platform.system()
        print('Operating System: ', classifier.OsName)
        print('LOOCV Fusion')

        classifier.test_name = test_name
        classifier.aggregateVideoFeatures = False
        if args.use_train_test_val:
            classifier.datasets = ['training', 'validation', 'test']

        classifier.base_path = args.base_path
        classifier.base_path2 = args.base_path2
        classifier.features_file_filter = args.features_file_filter
        classifier.label_path = args.label_path

        # Send e-mail Process Start
        send_email_start(classifier, True)

        # train the model
        #classifier.trainModelFV_LOOCV_Fusion_old()
        classifier.trainModelFV_LOOCV_Fusion_old_gmm()


def perform_loocv_fusion(args):
    # Instantiate Classifier
    classifier = Classifier(classifier=SVC(kernel='linear'), #classifier=LinearSVC(random_state=0, C=args.c_parameter),
                            no_clusters=args.number_cluster,
                            #gmm_random_state=0, 89.24
                            #gmm_random_state=1, 87.09
                            #gmm_random_state=99, 87.94
                            #gmm_random_state=3, 87.09
                            #gmm_random_state=13, 86.02
                            #gmm_random_state=7, 87.09
                            #gmm_random_state=5, 86.02
                            #gmm_random_state=17, 87.09
                            gmm_random_state=0,
                            no_samples=None)

    classifier.OsName = platform.system()
    print('Operating System: ', classifier.OsName)
    print('LOOCV Fusion')

    classifier.test_name = args.test_name
    classifier.aggregateVideoFeatures = False
    if args.use_train_test_val:
        classifier.datasets = ['training', 'validation', 'test']

    classifier.base_path = args.base_path
    classifier.features_file_filter = args.features_file_filter
    classifier.label_path = args.label_path

    # Send e-mail Process Start
    send_email_start(classifier, True)

    # train the model
    classifier.trainModelFV_LOOCV_Fusion()


def perform_loocv_classifiers(args):
    # Instantiate Classifier
    classifier = Classifier(classifier=None,
                            gmm_random_state=99,
                            no_clusters=args.number_cluster,
                            no_samples=None)

    classifier.OsName = platform.system()
    print('Operating System: ', classifier.OsName)
    print('LOOCV Classification')

    classifier.test_name = args.test_name
    classifier.aggregateVideoFeatures = False
    if args.use_train_test_val:
        classifier.datasets = ['training', 'validation', 'test']

    classifier.base_path = args.base_path
    classifier.label_path = args.label_path

    # Send e-mail Process Start
    send_email_start(classifier)

    # train the model
    classifier.trainModelFV_LOOCV_Classifiers(extension=args.features_file_filter)


def send_email_start(classifier, send=True):
    # Send e-mail Process Start
    current_date = datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")
    classifier.parameters = 'Parameters\n'
    classifier.parameters += 'Test Name: %s\n' % classifier.test_name
    classifier.parameters += 'GMM Random State: %s\n' % classifier.gmm_random_state
    classifier.parameters += 'Base Path: %s\n' % classifier.base_path
    classifier.parameters += 'Base Path2: %s\n' % classifier.base_path2
    classifier.parameters += 'Train Path: %s\n' % classifier.train_path
    classifier.parameters += 'Test Path: %s\n' % classifier.test_path
    classifier.parameters += 'Label Path: %s\n' % classifier.label_path
    classifier.parameters += 'Scaler Cluster Type: %s\n' % classifier.scaler_type_cluster
    classifier.parameters += 'Scaler Type: %s\n' % classifier.scaler_type
    classifier.parameters += 'k: %s\n' % classifier.no_clusters
    classifier.parameters += 'Nr. Samples: %s\n' % classifier.no_samples
    print(classifier.parameters)
    if send:
        msg = "Process Start at: {}\n".format(current_date)
        msg += classifier.parameters
        classifier.mail_helper.sendMail(
            "Process Start: %s - %s" % (classifier.test_name, classifier.OsName), msg)


if __name__ == '__main__':
    main()
