"""Small argument checks shared by command-line entry points."""
import argparse


def nonempty(value):
    if not value.strip():
        raise argparse.ArgumentTypeError(
            'must not be empty; check the corresponding shell variable '
            '(for example VC_CFG, VC_TRAIN_RUN or VC_CKPT)')
    return value
