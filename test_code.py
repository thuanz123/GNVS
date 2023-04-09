from src.dataset import get_split_dataset


data_dir = '/lustre/scratch/client/vinai/users/tungdt33/data/cars'
train_set, val_set, test_set = get_split_dataset("srn", data_dir, want_split="all", training=True)
breakpoint()