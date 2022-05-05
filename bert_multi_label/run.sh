python data_processor.py

python run_classifier.py \
  --task_name news_label_29 \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --data_dir ../data/bert_multi_label_results/proc/ \
  --vocab_file pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 256 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --eval_batch_size 32 \
  --predict_batch_size 32 \
  --learning_rate 4e-3 \
  --num_train_epochs 3.0 \
  --export_model_dir ../data/bert_multi_label_results/epochs3/ \
  --output_dir ../data/bert_multi_label_results/epochs3/
#  --use_tpu=false \
#   --master='' \

# test bert
python run_test.py
#  --vocab_file pretrained_model/roberta_zh_l12/vocab.txt \
#  --bert_config_file pretrained_model/roberta_zh_l12/bert_config.json \
#  --init_checkpoint pretrained_model/roberta_zh_l12/bert_model.ckpt \