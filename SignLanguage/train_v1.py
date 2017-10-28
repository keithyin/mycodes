from torch import nn
from data.load_data import load_dataset, SignLangDataset, SignLangDataLoader
from data.some_script import id_2_sign
from itertools import count
from utils import routine
from utils import tools
from torch import optim
from nets.models import SimpleNN1DCNN
from tensorboardX import SummaryWriter


def main():
    root = "/home/fanyang/PycharmProjects/SignLanguage/data/tctodd"
    id_sign, sign_id = id_2_sign(root)

    for i in range(9):
        glos = globals()
        # initialize the record step

        glos['_record_step'] = 0

        glos['_record_epoch_step'] = 0

        train_samples, dev_samples = load_dataset(root, sign_id, hold_id=i)

        train_dataset = SignLangDataset(sample_list=train_samples)
        dev_dateset = SignLangDataset(sample_list=dev_samples)

        # model

        model = SimpleNN1DCNN()
        model.cuda()
        model.criterion = nn.CrossEntropyLoss()
        model.optimizer = optim.Adam(params=model.parameters())

        writer = SummaryWriter('ckpt/hold-id-%d' % i)

        for i in count():
            train_loader = SignLangDataLoader(dataset=train_dataset,
                                              batch_size=1, shuffle=True)
            dev_loader = SignLangDataLoader(dataset=dev_dateset,
                                            batch_size=1, shuffle=True)
            routine.train_one_epoch(train_loader, model=model, writer=writer)
            tools.adjust_learning_rate(model.optimizer)

            routine.validation(dev_loader, model=model, writer=writer)

            if i == 80:
                break


if __name__ == '__main__':
    main()
