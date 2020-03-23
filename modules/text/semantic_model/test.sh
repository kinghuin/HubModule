#!/bin/bash
for name in rbt3 rbtl3 \
bert_cased_L_12_H_768_A_12 bert_cased_L_24_H_1024_A_16 bert_chinese_L_12_H_768_A_12 bert_multi_cased_L_12_H_768_A_12 \
bert_multi_uncased_L_12_H_768_A_12 bert_uncased_L_12_H_768_A_12 bert_uncased_L_24_H_1024_A_16 chinese_bert_wwm \
chinese_bert_wwm_ext ernie ernie_tiny ernie_v2_eng_base ernie_v2_eng_large chinese_roberta_wwm_ext \
chinese-roberta-wwm-ext-large
  do
      python test_score_time.py --name $name
  done
