import sys
sys.path.append('byt5')
import paddle
import argparse
import sacrebleu

def parse_args():
    parser = argparse.ArgumentParser(
        description="getbleu"
    )
    parser.add_argument(
        "--allpred_path",
        type=str,
        default="/home/aistudio/data/data127876/",
        help="model_name_or_path",
    )
    
    args = parser.parse_args()

    return args


def bleu(targets, predictions):
  """Computes BLEU score.
  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings
  Returns:
    bleu_score across all targets and predictions
  """
  if isinstance(targets[0], list):
    targets = [[x for x in target] for target in targets]
  else:
    # Need to wrap targets in another list for corpus_bleu.
    targets = [targets]

  bleu_score = sacrebleu.corpus_bleu(predictions, targets,
                                     smooth_method="exp",
                                     smooth_value=0.0,
                                     force=False,
                                     lowercase=False,
                                     tokenize="intl",
                                     use_effective_order=False)
  return {"bleu": bleu_score.score}

def result(preddir):
    from paddlenlp.datasets import load_dataset
    ds = load_dataset("xsum",splits=("train", "dev","test"))
    datan = ds[1]
    datan = [data['label'] for data in datan]
    NOT_TRUNC_LABELS = datan
    pred = paddle.load(preddir)
    return bleu(NOT_TRUNC_LABELS,pred)


if __name__ == "__main__":
    args = parse_args()
    print(result(args.allpred_path))