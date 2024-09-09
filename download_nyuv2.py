from datasets import load_dataset

ds = load_dataset("sayakpaul/nyu_depth_v2")

ds.save_to_disk("../data/nyuv2")

print("caboo")