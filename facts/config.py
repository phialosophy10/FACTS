"""Dataset-wide constants for the FACTS dataset."""

# All 13 femurs in the dataset (3-digit IDs)
FEMURS = ["001", "002", "013", "015", "021", "026", "074",
          "075", "083", "086", "138", "164", "172"]

# Fixed test set (same as in the MIUA 2024 paper)
TEST_FEMURS = ["002", "086", "138"]
TRAIN_FEMURS = [f for f in FEMURS if f not in TEST_FEMURS]

# Isotropic voxel size of the (registered) volumes in micrometres
VOXEL_SIZE_UM = 58.0

# Synthetic degradation parameters
SYNTH_SIGMA = 1.2
SYNTH_SCALE = 4

# Default axis along which 2D slices are taken (consistent with earlier work,
# where slices were taken as vol[:, :, i])
SLICE_AXIS = 2

# Which file is used as network input / target for each data type.
# 'real'  : clinical CT (re-sliced to micro-CT grid)
# 'synth' : synthetic LR made from the histogram-matched micro-CT
INPUT_KEYS = {"real": ("LR", ""), "synth": ("SY", "hm")}
TARGET_KEY = ("HR", "hm")
MASK_KEY = ("MS", "")
