# coding:utf-8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import time

import paddlehub as hub

dataset = hub.dataset.ChnSentiCorp().get_dev_examples()
texts = [[sample.text_a, sample.text_b] for sample in dataset]
total = len(texts)
print("total= ", total)

with open("qps.txt", "w") as fout:
    fout.write("module\tuse_gpu\tbatch_size\tqps\n")
    for name in [
            "bert_cased_L_12_H_768_A_12", "bert_cased_L_24_H_1024_A_16",
            "bert_chinese_L_12_H_768_A_12", "bert_multi_cased_L_12_H_768_A_12",
            "bert_multi_uncased_L_12_H_768_A_12",
            "bert_uncased_L_12_H_768_A_12", "bert_uncased_L_24_H_1024_A_16",
            "bert_wwm_chinese_L_12_H_768_A_12",
            "bert_wwm_ext_chinese_L_12_H_768_A_12", "ernie", "ernie_tiny",
            "ernie_v2_eng_base", "ernie_v2_eng_large",
            "roberta_wwm_ext_chinese_L_12_H_768_A_12",
            "roberta_wwm_ext_chinese_L_12_H_768_A_12_distillation",
            "roberta_wwm_ext_chinese_L_24_H_1024_A_16",
            "roberta_wwm_ext_chinese_L_24_H_1024_A_16_distillation"
    ]:
        os.system("bash %s/download.sh %s" % (name, name))
        if name not in [
                "bert_cased_L_12_H_768_A_12", "bert_cased_L_24_H_1024_A_16",
                "ernie", "ernie_tiny", "ernie_v2_eng_base",
                "ernie_v2_eng_large", "roberta_wwm_ext_chinese_L_12_H_768_A_12",
                "roberta_wwm_ext_chinese_L_12_H_768_A_12_distillation",
                "roberta_wwm_ext_chinese_L_24_H_1024_A_16",
                "roberta_wwm_ext_chinese_L_24_H_1024_A_16_distillation"
        ]:
            continue
        module = hub.Module(
            name=name.replace("L_12_H_768_A_12", "L-12_H-768_A-12").replace(
                "L_24_H_1024_A_16", "L-24_H-1024_A-16"))

        if name in [
                "bert_cased_L_24_H_1024_A_16",
                "bert_uncased_L_24_H_1024_A_16",
                "ernie_v2_eng_large",
                "roberta_wwm_ext_chinese_L_24_H_1024_A_16",
        ]:
            batch_sizes = [1, 4, 8]
        else:
            batch_sizes = [1, 8, 16]

        for use_gpu in [True]:
            for batch_size in batch_sizes:
                module.get_embedding(texts=[["hello"]], use_gpu=True)
                start = time.time()
                module.get_embedding(
                    texts=texts, use_gpu=use_gpu, batch_size=batch_size)
                use_time = time.time() - start
                qps = float(total) / float(use_time)
                print(
                    "%s with gpu=%s, batch_size=%s, use %s seconds, its qps=%f "
                    % (name, use_gpu, batch_size, use_time, qps))
                fout.write(
                    "%s\t%s\t%s\t%s\n" % (name, use_gpu, batch_size, qps))
