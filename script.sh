#Train
python main.py \
  --config config/model-apl-ssn.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_apl_ssn_llfs_share_emo \
  --model model_apl_ssn.model_apl_ssn_llfs_share_emo.Model \
  --dataset model_apl_ssn.dataset_llfs.Dataset \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_M003_ssn_checkpoint_20.pt \
  --epochs 50 \
  --n_folders 10

#Eval
python main.py \
  --config config/model-apl-ssn.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix MEAD_apl_ssn_emo_wo_kdloss \
  --model model_apl_ssn.model_apl_ssn_emo_wo_kdloss.Model \
  --dataset model_apl_ssn.dataset_llfs.Dataset \
  --skip-train-val \
  --pretrained /home/cxnam/Documents/MyWorkingSpace/Trainer_copy/assets/checkpoints/best_MEAD_apl_ssn_emo_wo_kdloss_checkpoint_1_MSE=-0.4457.pt \
  --evaluation \
  --n_folders 10

