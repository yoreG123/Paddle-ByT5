<<<<<<< HEAD
import sys
sys.path.append('byt5')

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
    # get model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer=ByT5Tokenizer()

    # # get metric
    # hg_metric = hg_load_metric("rouge.py")
    # get dataloader
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

                if args.save_steps > 0 and (global_steps % args.save_steps == 0) :
                    logger.info("********** Running evaluating **********")
                    logger.info(f"********** Step {global_steps} **********")
                    output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    # rl,rb = evaluate(
                    #     model,
                    #     dev_dataloader,
                    #     tokenizer
                    # )
                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                logger.info("********** Running evaluating **********")
                logger.info(f"********** Step {global_steps} **********")
                output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                # rl,rb = evaluate(
                #     model,
                #     dev_dataloader,
                #     tokenizer
                # )
                logger.info("********** Evaluating Done **********")
                logger.info("********** Training Done **********")
                return


if __name__ == "__main__":
    args = parse_args()
=======
import sys
sys.path.append('byt5')

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
    # get model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer=ByT5Tokenizer()

    # # get metric
    # hg_metric = hg_load_metric("rouge.py")
    # get dataloader
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

                if args.save_steps > 0 and (global_steps % args.save_steps == 0) :
                    logger.info("********** Running evaluating **********")
                    logger.info(f"********** Step {global_steps} **********")
                    output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    # rl,rb = evaluate(
                    #     model,
                    #     dev_dataloader,
                    #     tokenizer
                    # )
                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                logger.info("********** Running evaluating **********")
                logger.info(f"********** Step {global_steps} **********")
                output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                # rl,rb = evaluate(
                #     model,
                #     dev_dataloader,
                #     tokenizer
                # )
                logger.info("********** Evaluating Done **********")
                logger.info("********** Training Done **********")
                return


if __name__ == "__main__":
    args = parse_args()
>>>>>>> 7c4a2eb02d18b33782184f66c6d771b296d4a929
    main(args)