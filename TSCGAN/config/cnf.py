import os
import sys

data_dir = '/home/fanyang/PycharmProjects/UCR_TS_Archive_2015'
proj_dir = sys.path[0]


class Adia(object):
    DATASET_NAME = 'Adiac'
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 16  # Number of filters in first conv layer
    num_filt_2 = 14  # Number of filters in second conv layer
    num_filt_3 = 8  # Number of filters in thirs conv layer
    num_fc_1 = 40  # Number of neurons in hully connected layer
    max_iterations = 20000
    BATCH_SIZE = 64
    dropout = 1.0  # Dropout rate in the fully connected layer
    plot_row = 5  # How many rows do you want to plot in the visualization
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 37
    FEATURE_LEN = 176


class MedicalImages1(object):
    DATASET_NAME = 'MedicalImages'
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 16  # Number of filters in first conv layer
    num_filt_2 = 14  # Number of filters in second conv layer
    num_filt_3 = 8  # Number of filters in thirs conv layer
    num_fc_1 = 40  # Number of neurons in hully connected layer
    max_iterations = 20000
    BATCH_SIZE = 64
    dropout = 1.0  # Dropout rate in the fully connected layer
    plot_row = 5  # How many rows do you want to plot in the visualization
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 10
    FEATURE_LEN = 99


class Elec(object):
    DATASET_NAME = 'Elec'
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30  # Number of filters in first conv layer 16
    num_filt_2 = 14  # Number of filters in second conv layer 14
    num_filt_3 = 12  # Number of filters in thirs conv layer 8
    num_fc_1 = 40  # Number of neurons in hully connected layer
    max_iterations = 20000
    BATCH_SIZE = 40
    dropout = 1.0  # Dropout rate in the fully connected layer
    plot_row = 5  # How many rows do you want to plot in the visualization
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 10
    FEATURE_LEN = 96
    num_train_data = 2193
    num_test_data = 727


class Gun_Point1(object):
    DATASET_NAME = "Gun_Point"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 14
    num_filt_2 = 10
    num_filt_3 = 8
    num_fc_1 = 20
    max_iterations = 20000
    BATCH_SIZE = 5
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 2
    FEATURE_LEN = 150
    num_train_data = 50
    num_test_data = 150


class Coffee(object):
    DATASET_NAME = "Coffee"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 14
    num_filt_2 = 10
    num_filt_3 = 8
    num_fc_1 = 20
    max_iterations = 20000
    BATCH_SIZE = 2
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 2
    FEATURE_LEN = 286
    num_train_data = 28
    num_test_data = 28


class Adiac(object):
    DATASET_NAME = "Coffee"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 40
    num_filt_2 = 14
    num_filt_3 = 12
    num_fc_1 = 40
    max_iterations = 20000
    BATCH_SIZE = 20
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 37
    FEATURE_LEN = 176
    num_train_data = 390
    num_test_data = 391


class Beef(object):
    DATASET_NAME = "Beef"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30
    num_filt_2 = 14
    num_filt_3 = 12
    num_fc_1 = 30
    max_iterations = 20000
    BATCH_SIZE = 3
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 5
    FEATURE_LEN = 470
    num_train_data = 30
    num_test_data = 30


class CBF(object):
    DATASET_NAME = "CBF"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 20
    num_filt_2 = 14
    num_filt_3 = 10
    num_fc_1 = 20
    max_iterations = 20000
    BATCH_SIZE = 3
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 3
    FEATURE_LEN = 128
    num_train_data = 30
    num_test_data = 900


class ChlorineCon(object):
    DATASET_NAME = "ChlorineConcentration"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30
    num_filt_2 = 16
    num_filt_3 = 12
    num_fc_1 = 30
    max_iterations = 20000
    BATCH_SIZE = 20
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 3
    FEATURE_LEN = 166
    num_train_data = 467
    num_test_data = 3840


class CinC_ECG_torso(object):
    DATASET_NAME = "CinC_ECG_torso"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30
    num_filt_2 = 14
    num_filt_3 = 12
    num_fc_1 = 30
    max_iterations = 20000
    BATCH_SIZE = 4
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 4
    FEATURE_LEN = 1639
    num_train_data = 40
    num_test_data = 1380


class Cricket_X(object):
    DATASET_NAME = "Cricket_X"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30
    num_filt_2 = 16
    num_filt_3 = 12
    num_fc_1 = 30
    max_iterations = 20000
    BATCH_SIZE = 20
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 12
    FEATURE_LEN = 300
    num_train_data = 390
    num_test_data = 390


class Cricket_Y(object):
    DATASET_NAME = "Cricket_Y"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30
    num_filt_2 = 16
    num_filt_3 = 12
    num_fc_1 = 60
    max_iterations = 20000
    BATCH_SIZE = 30
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 12
    FEATURE_LEN = 300
    num_train_data = 390
    num_test_data = 390


class Cricket_Z(object):
    DATASET_NAME = "Cricket_Z"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 20
    num_filt_2 = 12
    num_filt_3 = 10
    num_fc_1 = 30
    max_iterations = 20000
    BATCH_SIZE = 30
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 12
    FEATURE_LEN = 300
    num_train_data = 390
    num_test_data = 390


class DiatomSizeReduction(object):
    DATASET_NAME = "DiatomSizeReduction"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 20
    num_filt_2 = 14
    num_filt_3 = 8
    num_fc_1 = 20
    max_iterations = 20000
    BATCH_SIZE = 5
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 4
    FEATURE_LEN = 345
    num_train_data = 16
    num_test_data = 306


class ECGFiveDays(object):
    DATASET_NAME = "ECGFiveDays"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 14
    num_filt_2 = 12
    num_filt_3 = 8
    num_fc_1 = 22
    max_iterations = 20000
    BATCH_SIZE = 3
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 2
    FEATURE_LEN = 136
    num_train_data = 23
    num_test_data = 861


class FaceAll(object):
    DATASET_NAME = "FaceAll"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30
    num_filt_2 = 14
    num_filt_3 = 10
    num_fc_1 = 40
    max_iterations = 20000
    BATCH_SIZE = 30
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 14
    FEATURE_LEN = 131
    num_train_data = 560
    num_test_data = 1690


class FaceFour(object):
    DATASET_NAME = "FaceFour"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 20
    num_filt_2 = 12
    num_filt_3 = 10
    num_fc_1 = 24
    max_iterations = 20000
    BATCH_SIZE = 3
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 4
    FEATURE_LEN = 350
    num_train_data = 24
    num_test_data = 88


class FacesUCR(object):
    DATASET_NAME = "FacesUCR"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 40
    num_filt_2 = 14
    num_filt_3 = 10
    num_fc_1 = 40
    max_iterations = 20000
    BATCH_SIZE = 20
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 14
    FEATURE_LEN = 131
    num_train_data = 200
    num_test_data = 2050


class FISH(object):
    DATASET_NAME = "FISH"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30
    num_filt_2 = 14
    num_filt_3 = 12
    num_fc_1 = 30
    max_iterations = 20000
    BATCH_SIZE = 16
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 7
    FEATURE_LEN = 463
    num_train_data = 175
    num_test_data = 175


class Gun_Point(object):
    DATASET_NAME = "Gun_Point"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 20
    num_filt_2 = 12
    num_filt_3 = 10
    num_fc_1 = 20
    max_iterations = 20000
    BATCH_SIZE = 6
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 2
    FEATURE_LEN = 150
    num_train_data = 50
    num_test_data = 150


class Haptics(object):
    DATASET_NAME = "Haptics"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 10
    num_filt_2 = 20
    num_filt_3 = 30
    num_fc_1 = 40
    max_iterations = 20000
    BATCH_SIZE = 15
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 5
    FEATURE_LEN = 1092
    num_train_data = 155
    num_test_data = 308



class ItalyPower(object):
    DATASET_NAME = "ItalyPowerDemand"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30
    num_filt_2 = 15
    num_filt_3 = 10
    num_fc_1 = 20
    max_iterations = 20000
    BATCH_SIZE = 6
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 2
    FEATURE_LEN = 24
    num_train_data = 67
    num_test_data = 1029


class Lighting7(object):
    DATASET_NAME = "Lighting7"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 20
    num_filt_2 = 14
    num_filt_3 = 10
    num_fc_1 = 30
    max_iterations = 20000
    BATCH_SIZE = 6
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 7
    FEATURE_LEN = 319
    num_train_data = 70
    num_test_data = 73


class MedicalImages(object):
    DATASET_NAME = "MedicalImages"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 40
    num_filt_2 = 20
    num_filt_3 = 14
    num_fc_1 = 40
    max_iterations = 20000
    BATCH_SIZE = 20
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 10
    FEATURE_LEN = 99
    num_train_data = 381
    num_test_data = 760


class MoteStrain(object):
    DATASET_NAME = "MoteStrain"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 20
    num_filt_2 = 14
    num_filt_3 = 12
    num_fc_1 = 20
    max_iterations = 20000
    BATCH_SIZE = 3
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 2
    FEATURE_LEN = 84
    num_train_data = 20
    num_test_data = 1252


class NonInvasiveFatalECG_Thorax1(object):
    DATASET_NAME = "NonInvasiveFatalECG_Thorax1"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 4
    num_filt_2 = 8
    num_filt_3 = 16
    num_fc_1 = 100
    max_iterations = 20000
    BATCH_SIZE = 32
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 42
    FEATURE_LEN = 750
    num_train_data = 1800
    num_test_data = 1965
    # 80 14 12 60 batch=64 .85
    # 80 14 12 40 batch=32 .86
    # 32 64 128 1000 batch=32 .86
    # 4 8 16 100 batch=32 .82


class OliveOil(object):
    DATASET_NAME = "OliveOil"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 40
    num_filt_2 = 14
    num_filt_3 = 12
    num_fc_1 = 30
    max_iterations = 20000
    BATCH_SIZE = 4
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 4
    FEATURE_LEN = 570
    num_train_data = 30
    num_test_data = 30


class Yoga(object):
    DATASET_NAME = "yoga"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 30
    num_filt_2 = 20
    num_filt_3 = 14
    num_fc_1 = 20
    max_iterations = 20000
    BATCH_SIZE = 24
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 2
    FEATURE_LEN = 426
    num_train_data = 300
    num_test_data = 3000


class Lighting2(object):
    DATASET_NAME = "Lighting2"
    TRAIN_RECORDS = os.path.join(proj_dir, "data/%s_train.tfrecords" % DATASET_NAME)
    TEST_RECORDS = os.path.join(data_dir, DATASET_NAME + '/%s_TEST' % DATASET_NAME)
    num_filt_1 = 2
    num_filt_2 = 4
    num_filt_3 = 4
    num_fc_1 = 10
    max_iterations = 20000
    BATCH_SIZE = 16
    dropout = 1.0
    plot_row = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 2
    FEATURE_LEN = 637
    num_train_data = 60
    # 4 8 16 100 batch=32 .63
