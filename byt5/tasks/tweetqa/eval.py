import sys
sys.path.append('byt5')

import re
import string
import logging
import nltk
import paddle
from filelock import FileLock
from paddlenlp.datasets import load_dataset
from data import get_dev_dataloader
from tqdm.auto import tqdm
from paddlenlp.transformers import T5ForConditionalGeneration
from paddlenlp.transformers import ByT5Tokenizer
import argparse
import numpy as np
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import sentence_bleu

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)

def normalize_text(text):
    """Normalize text for TweetQA task.
    Args:
        text: string
    Returns:
        normalized string
    """
    # Lower case.
    text = text.lower()
    # Remove punctuation.
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    # Remove articles.
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Fix extra whitespaces.
    text = ' '.join(text.split())
    return text

def bleu1(targets, predictions):
  """BLEU-1 with normalized targets and predictions.
  Code has been adapted from tweetqa_eval.py since we want this BLEU-1
  calculation to be identical to that used in the TweetQA paper.
  Args:
    targets: list of strings.
    predictions: list of strings.
  Returns:
    A dictionary that looks like: {"bleu": <value>}
  """
  bleu_scores = []
  for target, prediction in zip(targets, predictions):
    target = normalize_text(target)
    prediction = normalize_text(prediction)

    # By setting the weights tuple to be (1, 0, 0, 0), only uni-grams are
    # counted towards the BLEU score, resulting in BLEU-1.
    score = sentence_bleu(
        [target.split()], prediction.split(), weights=(1, 0, 0, 0)) * 100

    bleu_scores.append(score)

  return {'bleu1': np.mean(bleu_scores)}

def compute(predictions, references, rouge_types=None, use_agregator=True,use_stemmer=False):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
        if use_agregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            if use_agregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)

        if use_agregator:
            result = aggregator.aggregate()
        else:
            result = {}
            for key in scores[0]:
                result[key] = list(score[key] for score in scores)

        return result

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ByT5 model on a summarize task"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/aistudio/byt5/tasks/tweetqa/tqoutputs/stepf-18450",
        help="model_name_or_path",
    )
    parser.add_argument(
        "--evaluate_file",
        type=str,
        default="/home/aistudio/datasets/tweetqa/dev.json",
        help="evaluate_file.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=32,
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
        '--faster',
        action='store_true',
        help='Whether to process inference using faster transformer. ',
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
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

def main():
    args = parse_args()

    def postprocess_text(datas):
        datas = [data for data in datas]
        return datas

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.eval()

    tokenizer = ByT5Tokenizer()
    eval_dataloader = get_dev_dataloader(tokenizer, args)
    ds = load_dataset("tweetqa_paddle",splits=("train", "dev","test"))
    datan = ds[1]
    datan = [data['label'][0] for data in datan]
    NOT_TRUNC_LABELS = datan

    gen_kwargs = {
        "max_length": args.max_target_length - 1,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
        # "decode_strategy": "beam_search",
        "decode_strategy": "greedy_search",
        "top_k":10,
        # # "use_faster":args.faster
        "top_p":0.7
    }


    all_preds = []
    with paddle.no_grad():
        for batch in tqdm(eval_dataloader):
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
    # for pred, label in zip(all_preds, NOT_TRUNC_LABELS):
    #     rougel.add_inst(pred, label)
    #     bleu.add_inst(pred, label)
    # resultl = rougel.score()
    # resultb = bleu.score()
    results = compute([normalize_text(data) for data in all_preds],[normalize_text(data) for data in NOT_TRUNC_LABELS])
    # print(results)
    print('#######################################')
    print('rogueL:',results["rougeL"].mid.fmeasure*100)
    print (bleu1([normalize_text(data) for data in NOT_TRUNC_LABELS],[normalize_text(data) for data in all_preds])['bleu1'])
    paddle.save(all_preds, "all_preds.pd")

if __name__ == "__main__":
    main()
