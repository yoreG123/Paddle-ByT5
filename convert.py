from collections import OrderedDict
import argparse

dont_transpose = [
    "shared.weight",
    "layer_norm.weight",
    ".layer_norm.weight",
    "relative_attention_bias.weight",
    "embed_tokens.weight"
]


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path, paddle_dump_path):
    import torch
    import paddle

    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        transpose = False

        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    transpose = True
 
        print(f"Converting: {k} | is_transpose {transpose}")

        if k!="lm_head.weight":
            k = "t5." + k
        paddle_state_dict[k] = v.data.numpy()
    
    
    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default="google/byt5-small/pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the Pytorch checkpoint path.",
    )
    parser.add_argument(
        "--paddle_dump_path",
        default="paddle/byt5-small/model_state.pdparams",
        type=str,
        required=False,
        help="Path to the output Paddle model.",
    )
    args = parser.parse_args()
    convert_pytorch_checkpoint_to_paddle(
        args.pytorch_checkpoint_path, args.paddle_dump_path
    )