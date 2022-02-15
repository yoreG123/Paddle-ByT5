import sys
sys.path.append('byt5')

import re
import string
import logging
import math
import os
import nltk
from filelock import FileLock
import paddle
from paddle.amp import GradScaler, auto_cast
from paddle.optimizer import AdamW
from paddlenlp.transformers import T5ForConditionalGeneration
from tqdm import tqdm
import numpy as np
from args import parse_args
from paddlenlp.transformers import ByT5Tokenizer
from data import get_dev_dataloader, get_train_dataloader
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import sentence_bleu

from utils import (
    get_writer,
    save_json,
    set_seed,
)
from paddlenlp.datasets import load_dataset
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


logger = logging.getLogger(__name__)

def postprocess_text(datas):
    datas = [data for data in datas]
    return datas


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


@paddle.no_grad()
def evaluate(
    model, data_loader, tokenizer
):
    model.eval()

    gen_kwargs = {
      "early_stopping": True,
      "length_penalty": 0.6,
      "max_length": 128,
      "min_length": 0,
    #   "num_beams": 4
      "decode_strategy": "greedy_search",
      "top_k":2,
      "top_p":0.9
    }
    all_preds = []
    ds = load_dataset("tweetqa_paddle",splits=("train", "dev","test"))
    datan = ds[1]
    datan = [data['label'][0] for data in datan]
    NOT_TRUNC_LABELS = datan
    for batch in tqdm(data_loader):
        source_ids, source_mask, labels, target_mask = batch
        generated_tokens = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            **gen_kwargs,
        )[0]
        labels = np.where(labels.numpy() != -100, labels.numpy(), tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(generated_tokens.numpy(), skip_special_tokens=True)
        decoded_preds = postprocess_text(decoded_preds)
        all_preds.extend(decoded_preds)
    # print(all_preds)
    # print(NOT_TRUNC_LABELS)
    results = compute([normalize_text(data) for data in all_preds],[normalize_text(data) for data in NOT_TRUNC_LABELS])
    print('rogueL:',results["rougeL"].mid.fmeasure)
    bleu1r = bleu1([normalize_text(data) for data in NOT_TRUNC_LABELS],[normalize_text(data) for data in all_preds])
    print (bleu1r)
    return results["rougeL"].mid.fmeasure,bleu1r['bleu1']


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "run.log"),
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")
    paddle.set_device(args.device)
    set_seed(args)

    writer = get_writer(args)

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer=ByT5Tokenizer()

    train_dataloader = get_train_dataloader(tokenizer, args)
    dev_dataloader = get_dev_dataloader(tokenizer, args)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps > 0:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
    else:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    decay_params = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    optimizer = AdamW(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=args.max_grad_norm)
    )

    if args.use_amp:
        scaler = GradScaler(init_loss_scaling=args.scale_loss)

    logger.info("********** Running training **********")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous train batch size = {args.train_batch_size}")
    logger.info(f"  Instantaneous eval batch size = {args.eval_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    save_json(vars(args), os.path.join(args.output_dir, "args.json"))
    progress_bar = tqdm(range(args.max_train_steps))

    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0

    for _ in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            with auto_cast(
                args.use_amp, custom_white_list=["layer_norm", "softmax", "gelu"]
            ):
                source_ids, source_mask, labels, target_mask = batch
                outputs = model(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    labels=labels,
                    decoder_attention_mask=target_mask,
                )
                loss = outputs[0] / args.gradient_accumulation_steps
                tr_loss += loss.item()

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                if args.use_amp:
                    scaler.minimize(optimizer, loss)
                else:
                    optimizer.step()

                optimizer.clear_grad()
                progress_bar.update(1)
                global_steps += 1
                writer.add_scalar(
                        "train_loss",
                        loss.item(),
                        global_steps,
                    )
                print(loss.item())
                if args.logging_steps > 0 and global_steps % args.logging_steps == 0 :
                    writer.add_scalar("lr", args.learning_rate, global_steps)
                    writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_steps,
                    )
                    logger.info(
                        "global_steps {} - lr: {:.10f}  loss: {:.8f}".format(
                            global_steps,
                            args.learning_rate,
                            (tr_loss - logging_loss) / args.logging_steps,
                        )
                    )
                    logging_loss = tr_loss

                if args.save_steps > 0 and (global_steps % args.save_steps == 0):
                    logger.info("********** Running evaluating **********")
                    logger.info(f"********** Step {global_steps} **********")
                    output_dir = os.path.join(args.output_dir, f"stepf-{global_steps}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    model.save_pretrained(output_dir)
                    rl,rb = evaluate(
                        model,
                        dev_dataloader,
                        tokenizer
                    )
                    writer.add_scalar("rougueL", rl, global_steps)
                    logger.info(f"rougueL = {rl}")
                    writer.add_scalar(f"bleu1", rb, global_steps)
                    logger.info(f"bleu1 = {rb}")
                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                logger.info("********** Running evaluating **********")
                logger.info(f"********** Step {global_steps} **********")
                output_dir = os.path.join(args.output_dir, f"stepf-{global_steps}")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                rl,rb = evaluate(
                    model,
                    dev_dataloader,
                    tokenizer
                )
                writer.add_scalar("rougeL", rl, global_steps)
                logger.info(f"rougeL = {rl}")
                writer.add_scalar(f"bleu1", rb, global_steps)
                logger.info(f"bleu1 = {rb}")
                logger.info("********** Evaluating Done **********")
                logger.info("********** Training Done **********")
                return


if __name__ == "__main__":
    args = parse_args()
    main(args)