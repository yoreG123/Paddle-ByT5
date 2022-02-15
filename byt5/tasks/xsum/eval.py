<<<<<<< HEAD
import sys
sys.path.append('byt5')

import logging
import nltk
import paddle
from filelock import FileLock
from paddle.io import Dataset
from paddlenlp.metrics import RougeL,BLEU
from paddlenlp.datasets import load_dataset
from data import get_dev_dataloader
from tqdm.auto import tqdm
from paddlenlp.transformers import T5ForConditionalGeneration
from paddlenlp.transformers import ByT5Tokenizer
import argparse
import numpy as np

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ByT5 model on a summarize task"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/aistudio/data/data127876/",
        help="model_name_or_path",
    )
    parser.add_argument(
        "--evaluate_file",
        type=str,
        default="/home/aistudio/data/data122619/dev.json",
        help="evaluate_file.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="summarize: ",
        help="A prefix to add before every source text ",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=4000,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        '--decode_strategy',
        default='greedy_search',
        type=str,
        help='The decode strategy in generation.')
    parser.add_argument(
        '--top_k',
        default=2,
        type=int,
        help='The number of highest probability vocabulary tokens to keep for top-k sampling.'
    )
    parser.add_argument(
        '--top_p',
        default=1.0,
        type=float,
        help='The cumulative probability for top-p sampling.')
    parser.add_argument(
        "--num_beams",
        type=int,
        default=16,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=2.0,
        help="length_penalty",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers.",
    )
    args = parser.parse_args()

    return args

# #########################################################

# def bleu(targets, predictions):
#   """Computes BLEU score.
#   Args:
#     targets: list of strings or list of list of strings if multiple references
#       are present.
#     predictions: list of strings
#   Returns:
#     bleu_score across all targets and predictions
#   """
#   if isinstance(targets[0], list):
#     targets = [[x for x in target] for target in targets]
#   else:
#     # Need to wrap targets in another list for corpus_bleu.
#     targets = [targets]

#   bleu_score = sacrebleu.corpus_bleu(predictions, targets,
#                                      smooth_method="exp",
#                                      smooth_value=0.0,
#                                      force=False,
#                                      lowercase=False,
#                                      tokenize="intl",
#                                      use_effective_order=False)
#   return {"bleu": bleu_score.score}
# ##############################################################
def main():
    args = parse_args()
    print(args)
    # def postprocess_text(datas):
    #     datas = [data.strip() for data in datas]
    #     datas = ["\n".join(nltk.sent_tokenize(data[2:-3])) for data in datas]
    #     return datas

    def postprocess_text(datas):
        datas = [data for data in datas]
        return datas

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.eval()

    tokenizer = ByT5Tokenizer()
    eval_dataloader = get_dev_dataloader(tokenizer, args)
    ds = load_dataset("xsum",splits=("train", "dev","test"))
    datan = ds[1]
    datan = [[data['label']] for data in datan]
    NOT_TRUNC_LABELS = datan

    gen_kwargs = {
        "max_length": args.max_target_length - 1,
        # "min_length": 50,
        # "min_length": 0,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
        "decode_strategy": "beam_search"
        # "decode_strategy":args.decode_strategy,
        # "top_k":args.top_k,
        # "top_p":args.top_p,
    }
    # rougel = RougeL()
    # bleu = BLEU()


    all_preds = []
    with paddle.no_grad():
        i = 0
        for batch in tqdm(eval_dataloader):
            i+=1
            # if i>5:
            #     break
            logger.info(f"batch{i}_data")
            source_ids, source_mask, labels, target_mask = batch
            generated_tokens = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                **gen_kwargs,
            )[0]
            labels = np.where(labels.numpy() != -100, labels.numpy(), tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(
                generated_tokens.numpy(), skip_special_tokens=True
            )
            # print('decodep:',decoded_preds)
            decoded_preds = postprocess_text(decoded_preds)
            all_preds.extend(decoded_preds)
    # print(all_preds)
    # print(NOT_TRUNC_LABELS)
    # for pred, label in zip(all_preds, NOT_TRUNC_LABELS):
    #     rougel.add_inst(pred, label)
    #     bleu.add_inst(pred, label)
    # i = 0
    # for pred, label in zip(all_preds, NOT_TRUNC_LABELS):
    #     # print(pred,label)
    #     rougel.add_inst(pred, label)
    #     bleu.add_inst(pred, label)
    #     if i>100:
    #         break
    # resultl = rougel.score()
    # resultb = bleu.score()
    
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # result = {k: round(v, 4) for k, v in result.items()}

    # print('rougel:',resultl,'bleu:',resultb)
    # logger.info(f"{resultl}: {resultb}")
    paddle.save(all_preds, "all_preds.pd")

if __name__ == "__main__":
=======
import sys
sys.path.append('byt5')

import logging
import nltk
import paddle
from filelock import FileLock
from paddle.io import Dataset
from paddlenlp.metrics import RougeL,BLEU
from paddlenlp.datasets import load_dataset
from data import get_dev_dataloader
from tqdm.auto import tqdm
from paddlenlp.transformers import T5ForConditionalGeneration
from paddlenlp.transformers import ByT5Tokenizer
import argparse
import numpy as np

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ByT5 model on a summarize task"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/aistudio/data/data127876/",
        help="model_name_or_path",
    )
    parser.add_argument(
        "--evaluate_file",
        type=str,
        default="/home/aistudio/data/data122619/dev.json",
        help="evaluate_file.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="summarize: ",
        help="A prefix to add before every source text ",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=4000,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        '--decode_strategy',
        default='greedy_search',
        type=str,
        help='The decode strategy in generation.')
    parser.add_argument(
        '--top_k',
        default=2,
        type=int,
        help='The number of highest probability vocabulary tokens to keep for top-k sampling.'
    )
    parser.add_argument(
        '--top_p',
        default=1.0,
        type=float,
        help='The cumulative probability for top-p sampling.')
    parser.add_argument(
        "--num_beams",
        type=int,
        default=16,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=2.0,
        help="length_penalty",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers.",
    )
    args = parser.parse_args()

    return args

# #########################################################

# def bleu(targets, predictions):
#   """Computes BLEU score.
#   Args:
#     targets: list of strings or list of list of strings if multiple references
#       are present.
#     predictions: list of strings
#   Returns:
#     bleu_score across all targets and predictions
#   """
#   if isinstance(targets[0], list):
#     targets = [[x for x in target] for target in targets]
#   else:
#     # Need to wrap targets in another list for corpus_bleu.
#     targets = [targets]

#   bleu_score = sacrebleu.corpus_bleu(predictions, targets,
#                                      smooth_method="exp",
#                                      smooth_value=0.0,
#                                      force=False,
#                                      lowercase=False,
#                                      tokenize="intl",
#                                      use_effective_order=False)
#   return {"bleu": bleu_score.score}
# ##############################################################
def main():
    args = parse_args()
    print(args)
    # def postprocess_text(datas):
    #     datas = [data.strip() for data in datas]
    #     datas = ["\n".join(nltk.sent_tokenize(data[2:-3])) for data in datas]
    #     return datas

    def postprocess_text(datas):
        datas = [data for data in datas]
        return datas

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.eval()

    tokenizer = ByT5Tokenizer()
    eval_dataloader = get_dev_dataloader(tokenizer, args)
    ds = load_dataset("xsum",splits=("train", "dev","test"))
    datan = ds[1]
    datan = [[data['label']] for data in datan]
    NOT_TRUNC_LABELS = datan

    gen_kwargs = {
        "max_length": args.max_target_length - 1,
        # "min_length": 50,
        # "min_length": 0,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
        "decode_strategy": "beam_search"
        # "decode_strategy":args.decode_strategy,
        # "top_k":args.top_k,
        # "top_p":args.top_p,
    }
    # rougel = RougeL()
    # bleu = BLEU()


    all_preds = []
    with paddle.no_grad():
        i = 0
        for batch in tqdm(eval_dataloader):
            i+=1
            # if i>5:
            #     break
            logger.info(f"batch{i}_data")
            source_ids, source_mask, labels, target_mask = batch
            generated_tokens = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                **gen_kwargs,
            )[0]
            labels = np.where(labels.numpy() != -100, labels.numpy(), tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(
                generated_tokens.numpy(), skip_special_tokens=True
            )
            # print('decodep:',decoded_preds)
            decoded_preds = postprocess_text(decoded_preds)
            all_preds.extend(decoded_preds)
    # print(all_preds)
    # print(NOT_TRUNC_LABELS)
    # for pred, label in zip(all_preds, NOT_TRUNC_LABELS):
    #     rougel.add_inst(pred, label)
    #     bleu.add_inst(pred, label)
    # i = 0
    # for pred, label in zip(all_preds, NOT_TRUNC_LABELS):
    #     # print(pred,label)
    #     rougel.add_inst(pred, label)
    #     bleu.add_inst(pred, label)
    #     if i>100:
    #         break
    # resultl = rougel.score()
    # resultb = bleu.score()
    
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # result = {k: round(v, 4) for k, v in result.items()}

    # print('rougel:',resultl,'bleu:',resultb)
    # logger.info(f"{resultl}: {resultb}")
    paddle.save(all_preds, "all_preds.pd")

if __name__ == "__main__":
>>>>>>> 7c4a2eb02d18b33782184f66c6d771b296d4a929
    main()