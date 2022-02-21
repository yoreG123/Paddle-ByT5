# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['xsum']


class xsum(DatasetBuilder):

    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    # SPLITS = {
    #     'train': META_INFO(
    #         os.path.join('train.json'), 'BC2DA7633D04264CC3E53E67357269D7',
    #         'https://aistudio.baidu.com/bdvgpu/user/941056/3245322/files/tweetqa/train.json?_xsrf=2%7C0624a3bb%7Cb69c518918af24a9847b6af047618cdc%7C1638951933'),
    #     'dev': META_INFO(
    #         os.path.join('dev.json'), '0A385DF20E8B73FE5001659FAFF2F9D2',
    #         'https://aistudio.baidu.com/bdvgpu/user/941056/3245322/files/tweetqa/dev.json?_xsrf=2%7C0624a3bb%7Cb69c518918af24a9847b6af047618cdc%7C1638951933'),
    #     'test': META_INFO(
    #         os.path.join('test.json'), '7193F902C871468761B171B3155A89FC',
    #         'https://aistudio.baidu.com/bdvgpu/user/941056/3245322/files/tweetqa/test.json?_xsrf=2%7C0624a3bb%7Cb69c518918af24a9847b6af047618cdc%7C1638951933')
    # }
    SPLITS = {
        'train':"/home/aistudio/datasets/xsum/train.json" ,
        'dev': "/home/aistudio/datasets/xsum/dev.json",
        'test': "/home/aistudio/datasets/xsum/test.json"
    }

    def _get_data(self, mode, **kwargs):
        fullname = self.SPLITS[mode]
        return fullname

    def _read(self, filename, split):
        with open(filename, encoding="utf-8") as f:
            xsums = json.load(f)
            for data in xsums:
                yield {"document": data["document"], "label": data["summary"],"id": data["id"]}
