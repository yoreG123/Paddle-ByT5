<<<<<<< HEAD
import sys
sys.path.append('byt5')

import os
from functools import partial


from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset

from utils import load_pickle, save_pickle


def trans_func(example,tokenizer,args):
    inputs = 'question: %s context: %s' % (example["Question"], example["Tweet"])
    targets = example["label"][0]

    source = tokenizer(inputs, max_seq_len=args.max_source_length, pad_to_max_seq_len=False,return_attention_mask=True, return_token_type_ids=False,truncation_strategy="longest_first")

    target = tokenizer(targets, max_seq_len=args.max_target_length, pad_to_max_seq_len=False, return_attention_mask=True,return_token_type_ids=False,truncation_strategy="longest_first")

    return (
        source["input_ids"],
        source["attention_mask"],
        target["input_ids"],
        target["attention_mask"],
    )


def get_train_dataloader(tokenizer, args):
    filename = os.path.join("caches", "tweetqa_train" + ".pkl")
    print(filename)
    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("tweetqa_paddle", splits="train")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_dev_dataloader(tokenizer, args):
    filename = os.path.join("caches", "tweetqa" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("tweetqa_paddle", splits="dev")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.eval_batch_size, shuffle=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

=======
import sys
sys.path.append('byt5')

import os
from functools import partial


from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset

from utils import load_pickle, save_pickle


def trans_func(example,tokenizer,args):
    inputs = 'question: %s context: %s' % (example["Question"], example["Tweet"])
    targets = example["label"][0]

    source = tokenizer(inputs, max_seq_len=args.max_source_length, pad_to_max_seq_len=False,return_attention_mask=True, return_token_type_ids=False,truncation_strategy="longest_first")

    target = tokenizer(targets, max_seq_len=args.max_target_length, pad_to_max_seq_len=False, return_attention_mask=True,return_token_type_ids=False,truncation_strategy="longest_first")

    return (
        source["input_ids"],
        source["attention_mask"],
        target["input_ids"],
        target["attention_mask"],
    )


def get_train_dataloader(tokenizer, args):
    filename = os.path.join("caches", "tweetqa_train" + ".pkl")
    print(filename)
    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("tweetqa_paddle", splits="train")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_dev_dataloader(tokenizer, args):
    filename = os.path.join("caches", "tweetqa" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("tweetqa_paddle", splits="dev")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.eval_batch_size, shuffle=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

>>>>>>> 7c4a2eb02d18b33782184f66c6d771b296d4a929
    return data_loader