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

__all__ = ['Poetry']


class Poetry(DatasetBuilder):

    META_INFO = collections.namedtuple('META_INFO', ('file', 'md5', 'URL'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('train.json'), 'BC2DA7633D04264CC3E53E67357269D7',
            'https://aistudio.baidu.com/bdvgpu/user/941056/3245322/files/tweetqa/train.json?_xsrf=2%7C0624a3bb%7Cb69c518918af24a9847b6af047618cdc%7C1638951933'),
        'dev': META_INFO(
            os.path.join('dev.json'), '0A385DF20E8B73FE5001659FAFF2F9D2',
            'https://aistudio.baidu.com/bdvgpu/user/941056/3245322/files/tweetqa/dev.json?_xsrf=2%7C0624a3bb%7Cb69c518918af24a9847b6af047618cdc%7C1638951933'),
        'test': META_INFO(
            os.path.join('test.json'), '7193F902C871468761B171B3155A89FC',
            'https://aistudio.baidu.com/bdvgpu/user/941056/3245322/files/tweetqa/test.json?_xsrf=2%7C0624a3bb%7Cb69c518918af24a9847b6af047618cdc%7C1638951933')
    }

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, URL = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(URL, default_root)

        return fullname

    def _read(self, filename, *args):
        with open(filename, encoding="utf-8") as f:
            tweet_qa = json.load(f)
            i=100000
            for data in tweet_qa:
                id_ = data["qid"]+str(i)
                i+=1
                yield id_, {
                    "Question": data["Question"],
                    "Answer": [] if split == "test" else data["Answer"],
                    "Tweet": data["Tweet"],
                    "qid": data["qid"],
                }