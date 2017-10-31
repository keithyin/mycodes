def load_data_tttttttttttttt():
    from data.load_data import load_dataset, SignLangDataset, SignLangDataLoader
    from data.some_script import id_2_sign
    root = "/home/fanyang/PycharmProjects/SignLanguage/data/tctodd"
    file_name = "/home/fanyang/PycharmProjects/SignLanguage/data/tctodd/tctodd1/alive-1.tsd"
    id_sign, sign_id = id_2_sign(root)
    # train_samples, dev_samples = load_dataset(root, sign_id)

    for i in range(9):
        load_dataset(root, sign_id, hold_id=-i)


    # print(train_samples)
    # train_dataset = SignLangDataset(sample_list=train_samples)
    # loader = SignLangDataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    # iterrrr = iter(loader)
    # data, label = next(iterrrr)
    # print(data)
    # print(label)
    # for i, (batch_data, batch_label) in enumerate(loader):
    #     print(i, ":", batch_label.numpy()[0])


def ttttttttt_robot():
    from data.robot_data import load_data_robot
    train_dataset, dev_dataset = \
        load_data_robot("/home/fanyang/PycharmProjects/SignLanguage/data/robotfailuer/lp1.data.txt")
    print(len(train_dataset))
    print(len(dev_dataset))


def main():
    # load_data_tttttttttttttt()
    ttttttttt_robot()


if __name__ == '__main__':
    main()
