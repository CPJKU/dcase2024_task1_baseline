# Complexity Calculator for PyTorch models aligned with:
# https://github.com/AlbertoAncilotto/NeSsi/blob/main/nessi.py
# we only copy the complexity calculation for torch models from NeSsi to avoid
# including an additional tensorflow dependency in this code base

import torchinfo

MAX_PARAMS_MEMORY = 128_000
MAX_MACS = 30_000_000


def get_torch_size(model, input_size):
    model_profile = torchinfo.summary(model, input_size=input_size)
    return model_profile.total_mult_adds, model_profile.total_params
