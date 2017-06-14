import os


def generate_train_test_txt(data_root_dir, save_dir, rate=.1):
    train_txt_file_path = os.path.join(save_dir, "train.txt")
    test_txt_file_path = os.path.join(save_dir, "test.txt")
    dirs = os.listdir(data_root_dir)
    train_txt_file = open(train_txt_file_path, mode="w")
    test_txt_file = open(test_txt_file_path, mode="w")
    for i,level1 in enumerate(dirs):
        path_to_level1 = os.path.join(data_root_dir, level1)
        img_names = os.listdir(path_to_level1)

        test_num = rate*len(img_names)
        if test_num < 1:
            test_num = 1

        test_num = int(test_num)
        for img in img_names[:-test_num]:
            abs_path = os.path.join(level1, img)
            item = abs_path+" "+str(i)+"\n"
            train_txt_file.write(item)
        for img in img_names[-test_num:]:
            abs_path = os.path.join(level1, img)
            item = abs_path + " " + str(i) + "\n"
            test_txt_file.write(item)
        print("people ", i, " Done!")
    train_txt_file.close()
    test_txt_file.close()

def main():

    generate_train_test_txt("/home/dafu/PycharmProjects/data", save_dir="../data")


if __name__ == "__main__":
    main()
