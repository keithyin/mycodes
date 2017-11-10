import os


def generate_train_test_txt(data_root_dir, save_dir, rate=.1):
    train_txt_file_path = os.path.join(save_dir, "train.txt")
    test_txt_file_path = os.path.join(save_dir, "test.txt")
    dirs = os.listdir(data_root_dir)
    train_txt_file = open(train_txt_file_path, mode="w")
    test_txt_file = open(test_txt_file_path, mode="w")
    for i, level1 in enumerate(dirs):
        path_to_level1 = os.path.join(data_root_dir, level1)
        img_names = os.listdir(path_to_level1)

        test_num = rate * len(img_names)
        if test_num < 1:
            test_num = 1

        test_num = int(test_num)
        for img in img_names[:-test_num]:
            abs_path = os.path.join(level1, img)
            item = abs_path + " " + str(i) + "\n"
            train_txt_file.write(item)
        for img in img_names[-test_num:]:
            abs_path = os.path.join(level1, img)
            item = abs_path + " " + str(i) + "\n"
            test_txt_file.write(item)
        print("people ", i, " Done!")
    train_txt_file.close()
    test_txt_file.close()


def generate_txt(data_root_dir, save_dir, dataset_name='train'):
    txt_file_path = os.path.join(save_dir, dataset_name + '.txt')

    dirs = os.listdir(data_root_dir)
    dirs = sorted(dirs)
    txt_file = open(txt_file_path, mode="w")

    for i, level1 in enumerate(dirs):
        path_to_level1 = os.path.join(data_root_dir, level1)
        img_names = os.listdir(path_to_level1)
        for img in img_names:
            abs_path = os.path.join(level1, img)
            item = abs_path + " " + str(i) + "\n"
            txt_file.write(item)

        print("people ", i, " Done!")
    txt_file.close()


def person_id_to_folder(txt_file, save_dir):
    txt_file_path = os.path.join(save_dir, 'id2folder.txt')
    id_tracker = []
    with open(txt_file_path, mode='w') as target_file:
        with open(txt_file, mode="r") as file:
            try:
                for line in file:
                    img_desc = line.split(" ")

                    img_folder = img_desc[0].split("/")[0]
                    img_label = img_desc[1]
                    if int(img_label) not in id_tracker:
                        item = img_folder + " " + img_label
                        target_file.write(item)
                        id_tracker.append(int(img_label))

            except KeyboardInterrupt as e:
                file.close()
                target_file.close()


def get_seqlist_from_folder(folder):
    seqlist = []
    file_names = os.listdir(folder)
    file_names = sorted(file_names)
    seq_tracker = []
    sequence = None
    for file_name in file_names:
        camera = str(file_name[4:11])
        abs_file_name = os.path.join(folder, file_name)
        if camera not in seq_tracker:
            if sequence is not None:
                seqlist.append(sequence)
            seq_tracker.append(camera)
            sequence = []
        sequence.append(abs_file_name)
    seqlist.append(sequence)
    return seqlist


def main():
    # generate_train_test_txt("/media/fanyang/workspace/DataSet/MARS/bbox_train", save_dir="./")
    # generate_txt('/media/fanyang/workspace/DataSet/MARS/bbox_train',
    #              save_dir='./')
    generate_txt('/media/fanyang/workspace/DataSet/MARS/bbox_test',
                 save_dir='./', dataset_name='test_test')

    person_id_to_folder("./test_test.txt", save_dir="./")
    # get_seqlist_from_folder("/media/fanyang/workspace/DataSet/MARS/bbox_train/0001")


if __name__ == "__main__":
    main()
