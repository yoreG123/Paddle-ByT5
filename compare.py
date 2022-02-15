import sys
sys.path.append('byt5')
from paddlenlp.transformers import T5Model as PDT5Model
#  as PDT5Model
# from paddlenlp.transformers import ByT5Tokenizer
from transformers import T5Model as PTT5Model

import torch
import paddle

# paddle.set_device("cpu")

pd_model = PDT5Model.from_pretrained('byt5model')
pd_model.eval()
pt_model = PTT5Model.from_pretrained("google/byt5-small")
pt_model.eval()

with paddle.no_grad():
    pd_outputs = pd_model(
        **pd_model.dummy_inputs
    )[0]

with torch.no_grad():
    pt_outputs = pt_model(
        **pt_model.dummy_inputs
    ).last_hidden_state

def compare(a, b):
    a = torch.tensor(a.numpy()).float()
    b = torch.tensor(b.numpy()).float()
    meandif = (a - b).abs().mean()
    maxdif = (a - b).abs().max()
    print("mean difference:", meandif)
    print("max difference:", maxdif)


compare(pd_outputs, pt_outputs)