# ####################################################################
# reference file => https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
# bert embeddings utility file
# ####################################################################

import torch, os
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from uuid import uuid4
from time import time



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example"""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


def create_examples(df):
    """Creates examples for the provided sets.
    df with columns id & text"""
    examples = []
    for (i, row) in enumerate(df.values):
        # guid = str(uuid4()).replace("-", "")+str(time()).replace(".", "")
        # text_a = row[0]
        guid = row[0]
        text_a = row[1]
        labels = []
        examples.append(
            InputExample(guid=guid, text_a=text_a, labels=labels))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def get_bert_embeddings(docs_list, tokenizer, model, args):

    document = create_examples(docs_list)

    features = []
    for (ex_index, example) in enumerate(document):
        tokens_a = tokenizer.tokenize(str(example.text_a))
        #         tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(str(example.text_b))
            #             tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b,args["max_seq_length"] - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > args["max_seq_length"] - 2:
                tokens_a = tokens_a[:(args["max_seq_length"] - 2)]


        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (args["max_seq_length"] - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == args["max_seq_length"]
        assert len(input_mask) == args["max_seq_length"]
        assert len(segment_ids) == args["max_seq_length"]

        labels_ids = []
        for label in example.labels:
            labels_ids.append(float(label))

        #         label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=labels_ids))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.float)

    all_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    # eval_sampler = SequentialSampler(all_data)
    # eval_dataloader = DataLoader(all_data, sampler=eval_sampler, batch_size=args['batch_size'])
    dataloader = DataLoader(all_data, batch_size=args['batch_size'])

    all_logits = None
    all_labels = None

    model.eval()

    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(args["device"])
        input_mask = input_mask.to(args["device"])
        segment_ids = segment_ids.to(args["device"])
        label_ids = label_ids.to(args["device"])

        with torch.no_grad():
            # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)[1] # taking only the last layer embeddings

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)


    return all_logits

def get_single_doc_bert_embedding(single_doc, tokenizer, model):
    args = {
        "local_rank": -1,
        "no_cuda": False,
        "max_seq_length": 512,
        "batch_size": 32
    }

    if args["local_rank"] == -1 or args["no_cuda"]:
        args["device"] = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
        n_gpu = torch.cuda.device_count()
    #     n_gpu = 1
    else:
        torch.cuda.set_device(args['local_rank'])
        args["device"] = torch.device("cuda", args['local_rank'])
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    sample_df = pd.DataFrame([[0, single_doc]], columns=["filename", "text"])

    # # Load pre-trained model tokenizer (vocabulary)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # # # Loading pre-trained model (weights)
    # # # and putting the model in "evaluation" mode, meaning feed-forward operation.
    # model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.path.join("input_data", "bert"))
    if not args["no_cuda"]:
        model.to(args["device"])

    embeddings = get_bert_embeddings(sample_df, tokenizer, model, args)

    return list(embeddings[0])


if __name__ == "__main__":

    args = {
        "local_rank": -1,
        "no_cuda": False,
        "max_seq_length": 512,
        "batch_size": 32
    }

    if args["local_rank"] == -1 or args["no_cuda"]:
        args["device"] = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
        n_gpu = torch.cuda.device_count()
    #     n_gpu = 1
    else:
        torch.cuda.set_device(args['local_rank'])
        args["device"] = torch.device("cuda", args['local_rank'])
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    sample_df = pd.DataFrame([[0, "2 of our counterparties are writing letters of complaint.. here's a sample of some of the quotes we have heard from the 10 counterparties we have added in the last 6 months. "]], columns=["filename", "text"])

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # # Loading pre-trained model (weights)
    # # and putting the model in "evaluation" mode, meaning feed-forward operation.
    model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.path.join("input_data", "bert"))
    if not args["no_cuda"]:
        model.to(args["device"])

    embeddings = get_bert_embeddings(sample_df, tokenizer, model, args)

    embeddings1 =  get_signle_doc_bert_embedding("2 of our counterparties are writing letters of complaint.. here's a sample of some of the quotes we have heard from the 10 counterparties we have added in the last 6 months. ")

    assert list(embeddings[0]) == embeddings1