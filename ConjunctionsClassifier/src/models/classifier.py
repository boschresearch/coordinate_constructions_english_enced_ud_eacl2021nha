#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald


import torch
import code

from torch import nn

from models.wrappers import RobertaWrapper
from models.fc import FC
from data_handling.vocab import BasicVocab
from data_handling.dataset import ConjunctionPropagationDataset
from svm_utils import set_to_vec


class ConjunctionClassifier(nn.Module):
    """ Model for classifying dependency relations between tokens."""
    def __init__(self):
        super(ConjunctionClassifier, self).__init__()

        self.sent_embed = RobertaWrapper("../pretrained_models/roberta-base", fine_tune=False, output_dropout=0.0, token_mask_prob=0.0)
        self.dep_label_vocab = BasicVocab("../vocabs/dep_labels.txt")

        # Embeddings
        self.deprel_embed = nn.Embedding(len(self.dep_label_vocab), 50)
        self.in_out_embed = nn.Embedding(2, 50)
        self.lin_direction_embed = nn.Embedding(3, 50)
        self.already_has_dependent_embed = nn.Embedding(2, 50)

        self.fc = FC(3 * 768 + 50 + 50 + 50 + 50 + 2*len(self.dep_label_vocab) + 1, [1500, 500], 2, nn.ReLU(), p_dropout=0.33)

        # Uncomment for no tree
        # self.fc = FC(3 * 768 + 50 + 50, [1500, 500], 2, nn.ReLU(),
        #             p_dropout=0.33)

        # Uncomment for no token
        #self.fc = FC(50 + 50 + 50 + 50 + 2 * len(self.dep_label_vocab) + 1, [1500, 500], 2, nn.ReLU(),
        #             p_dropout=0.33)

    def forward(self, input_instances):
        # Embed sentences with BERT
        embeddings = self.sent_embed([instance.sentence for instance in input_instances])
        batch_size = embeddings.shape[0]
        seq_len = embeddings.shape[1]

        # Retrieve BERT embeddings according to the token indices specified by the instance
        conj_head_ixs = torch.cuda.LongTensor([inst.conj_head_ix for inst in input_instances])
        conj_dep_ixs = torch.cuda.LongTensor([inst.conj_dep_ix for inst in input_instances])
        candidate_ixs = torch.cuda.LongTensor([inst.candidate_ix for inst in input_instances])

        sent_num = torch.cuda.LongTensor(list(range(batch_size)))

        conj_head_embeddings = embeddings[sent_num, conj_head_ixs]
        conj_dep_embeddings = embeddings[sent_num, conj_dep_ixs]
        candidate_embeddings = embeddings[sent_num, candidate_ixs]

        assert len(conj_head_embeddings.shape) == len(conj_dep_embeddings.shape) == len(candidate_embeddings.shape) == 2
        assert conj_head_embeddings.shape[0] == conj_dep_embeddings.shape[0] == candidate_embeddings.shape[0] == batch_size

        # Retrieve dependency relation embeddings
        relation_embeddings = self.deprel_embed(torch.cuda.LongTensor([self.dep_label_vocab.token2ix[instance.relation_label] for instance in input_instances]))

        # Ingoing vs. outgoing (one-hot)
        in_out = self.in_out_embed(torch.cuda.LongTensor([0 if instance.dependency_direction == "incoming" else 1 for instance in input_instances]))

        # Linear direction of existing dependency vs. candidate propagated dependency
        lin_direction = self.lin_direction_embed(torch.cuda.LongTensor([0 if instance.linear_directions == "both-left" else \
                                                                   1 if instance.linear_directions == "both-right" else \
                                                                   2 for instance in input_instances]))

        # Already has dependent of same type?
        already_has_dependent = torch.stack([torch.cuda.FloatTensor([1]) if instance.already_has_dependent_type else torch.cuda.FloatTensor([0]) for instance in input_instances])
        already_has_dependent = self.already_has_dependent_embed(torch.cuda.LongTensor([1 if  instance.already_has_dependent_type else 0 for instance in input_instances]))

        # assert len(relation_embeddings.shape) == len(in_out.shape) == 2
        # assert relation_embeddings.shape[0] == in_out.shape[0] == batch_size

        # Outgoing dependency types of candidate governor and dependent
        outgoing_edges_gov = torch.stack([torch.cuda.FloatTensor(set_to_vec(instance.outgoing_edges_candidate_gov, self.dep_label_vocab)) for instance in input_instances])
        outgoing_edges_dep = torch.stack([torch.cuda.FloatTensor(set_to_vec(instance.outgoing_edges_candidate_dep, self.dep_label_vocab)) for instance in input_instances])

        # Number of items in coordination
        num_coord_items = torch.stack([torch.cuda.FloatTensor([float(instance.num_coord_items)]) for instance in input_instances])

        # Concatenate inputs
        concatenated_input = torch.cat((conj_head_embeddings, conj_dep_embeddings, candidate_embeddings,
                                        relation_embeddings, in_out, lin_direction, already_has_dependent,
                                        outgoing_edges_gov, outgoing_edges_dep, num_coord_items), dim=1)

        # Uncomment for no tree
        # concatenated_input = torch.cat((conj_head_embeddings, conj_dep_embeddings, candidate_embeddings, relation_embeddings, in_out), dim=1)

        # Uncomment for no token
        # concatenated_input = torch.cat((relation_embeddings, in_out, lin_direction, already_has_dependent,
        #                                outgoing_edges_gov, outgoing_edges_dep, num_coord_items), dim=1)

        # Run the fully connected network to get model outputs
        logits = self.fc(concatenated_input)

        # Return the result
        return logits

