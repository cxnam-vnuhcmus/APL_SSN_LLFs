#Train
python main.py \
  --config config/model-apl-ssn.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm \
  --suffix M003_apl_ssn_llfs_mel_wo_emo \
  --model model_apl_ssn.model_apl_ssn_llfs_mel_wo_emo.Model \
  --dataset model_apl_ssn.dataset_llfs_mel.Dataset \
  --skip-train-val \
  --epochs 300 \
  --pretrained ./assets/checkpoints/checkpoint_M003_apl_ssn_llfs_mel_wo_emo_checkpoint_164.pt \
  --n_folders 5 \
  

#Eval
python main.py \
  --config config/model-apl-ssn.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_apl_ssn_llfs_share_emo \
  --model model_apl_ssn.model_apl_ssn_llfs_share_emo.Model \
  --dataset model_apl_ssn.dataset_llfs.Dataset \
  --skip-train-val \
  --pretrained /home/cxnam/Documents/MyWorkingSpace/APL_SSN_LLFs/assets/checkpoints/best_M003_apl_ssn_llfs_share_emo_checkpoint_1_MSE=-0.6030.pt \
  --evaluation \
  --n_folders 10

python main.py \
  --config config/model-emo.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm \
  --suffix MEAD_emo_spec \
  --model model_apl_ssn.model_emo_spec.Model \
  --dataset model_apl_ssn.dataset_emo.Dataset \
  --skip-train-val \
  --n_folders 20
