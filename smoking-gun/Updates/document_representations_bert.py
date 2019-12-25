# ##############################################################################
# after process_data, now creating document represntations from bert embeddings
# ##############################################################################

import os, torch, logging
import pandas as pd
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel
from bert_embeddings import get_bert_embeddings


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_doc_representations(input_email_text_file, output_embedding_file, args, tokenizer, model):

    for ix, chunk_df in enumerate(pd.read_csv(input_email_text_file, chunksize=args["text_chunks"])):

        logging.info("embedding chunk number: {}".format(ix))

        embeddings = get_bert_embeddings(chunk_df, tokenizer, model, args)

        chunk_df["embedding"] = pd.Series(list(embeddings), index=chunk_df.index)

        with open(output_embedding_file, 'a', encoding="utf-8") as pf:
            chunk_df.to_csv(pf, header=False, index=False)

        # if ix > 1:
        #     break

    logging.info("Embeddings file successfully written to {}".format(output_embedding_file))

    return

def get_doc_representations(input_email_text_file = os.path.join("input_data", "email_texts.csv"),
                            output_embedding_file = os.path.join("input_data", "email_embeddings.csv")):
    embed_df = pd.DataFrame(columns=["filename", "text", "embedding"])
    embed_df.to_csv(output_embedding_file, index=False)

    args = {
        "local_rank": -1,
        "no_cuda": not torch.cuda.is_available(),
        "max_seq_length": 512,
        "batch_size": 32,
        "text_chunks": 124,
    }

    if args["local_rank"] == -1 or args["no_cuda"]:
        # for single cpu or single gpu unit
        args["device"] = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args['local_rank'])
        args["device"] = torch.device("cuda", args['local_rank'])
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # # Loading pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.path.join("input_data", "bert"))
    if not args["no_cuda"]:
        model.to(args["device"])

    batch_doc_representations(input_email_text_file, output_embedding_file, args, tokenizer, model)

    return


if __name__ == "__main__":

    get_doc_representations(input_email_text_file=os.path.join("input_data", "email_texts.csv"),
                            output_embedding_file=os.path.join("input_data", "email_embeddings.csv"))

    # input_email_text_file = os.path.join("input_data", "email_texts.csv")
    # output_embedding_file = os.path.join("input_data", "email_embeddings.csv")
    #
    # embed_df = pd.DataFrame(columns=["filename", "text", "embedding"])
    # embed_df.to_csv(output_embedding_file, index=False)
    #
    # args = {
    #     "local_rank": -1,
    #     "no_cuda": not torch.cuda.is_available(),
    #     "max_seq_length": 512,
    #     "batch_size": 32,
    #     "text_chunks": 124,
    # }
    #
    # if args["local_rank"] == -1 or args["no_cuda"]:
    #     # for single cpu or single gpu unit
    #     args["device"] = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    #     n_gpu = torch.cuda.device_count()
    # else:
    #     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args['local_rank'])
    #     args["device"] = torch.device("cuda", args['local_rank'])
    #     n_gpu = 1
    #     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.distributed.init_process_group(backend='nccl')
    #
    # # Load pre-trained model tokenizer (vocabulary)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # # # Loading pre-trained model (weights)
    # model = BertModel.from_pretrained('bert-base-uncased', cache_dir=os.path.join("input_data", "bert"))
    # if not args["no_cuda"]:
    #     model.to(args["device"])
    #
    # batch_doc_representations(input_email_text_file, output_embedding_file, args)
