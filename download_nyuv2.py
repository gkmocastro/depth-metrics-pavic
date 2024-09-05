from datasets import load_dataset

train_dataset = load_dataset("sayakpaul/nyu_depth_v2", download_mode="force_redownload")

train_dataset.save_to_disk("../data/nyuv2")

print("caboo")