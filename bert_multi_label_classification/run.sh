
python train.py \
  --epochs 3 \
  --class_nums 30 \
  --maxlen 512 \
  --batch_size 16 \
  --data_version ".all"


python predict.py

#  --model_path ".all" \
#  --data_version ".all"
