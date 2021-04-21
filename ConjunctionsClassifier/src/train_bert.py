#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald


from torch.utils.data import DataLoader

from models.classifier import ConjunctionClassifier
from data_handling.dataset import ConjunctionPropagationDataset
from trainer.trainer import Trainer

def main():
    model = ConjunctionClassifier()

    train_dataset = ConjunctionPropagationDataset.from_corpus_file("../propagation_train_no-aux.tsv")
    dev_dataset = ConjunctionPropagationDataset.from_corpus_file("../propagation_dev_no-aux.tsv")

    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda inst: inst)
    dev_data_loader = DataLoader(dev_dataset, batch_size=1, shuffle=True, collate_fn=lambda inst: inst)

    trainer = Trainer(model, train_data_loader, dev_data_loader)

    trainer.train()


if __name__ == '__main__':
    main()
