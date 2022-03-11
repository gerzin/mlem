import sys
import random
import string
import tempfile
from pathlib import Path

sys.path.append("../mlem")
import mlem.utilities as ut
import pandas as pd


def test_save_load_pickles():
    data_s = pd.DataFrame()
    data_s['Numbers'] = [*range(100)]
    data_s['Letters'] = [random.choice(string.ascii_letters) for _ in range(100)]
    with tempfile.TemporaryDirectory() as td:
        file = Path(td) / "example.bz2"
        ut.save_pickle_bz2(file, data_s)

        data_l = ut.load_pickle_bz2(file)

    assert (data_l == data_s).all().all()
