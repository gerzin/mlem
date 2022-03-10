import os
import sys
import tempfile
# ADD OTHER FOLDERS TO THIS LIST TO ADD THEM TO THE sys.path
from pathlib import Path

modules_to_add = ["", "mlem"]

this_file = os.path.abspath('')

for module in modules_to_add:
    p = Path(this_file).parent / module
    if p.exists():
        sys.path.append(str(p))
        print(f"ADDED: {p}")
    else:
        print(f"ERROR: {p} doesn't exist")

import mlem.models.classifiers as classifiers
from sklearn.datasets import make_multilabel_classification

feat, targ = make_multilabel_classification(n_samples=30, n_features=10, n_classes=4)


def test_classifier_creation():
    rfc = classifiers.MLEMRandomForestClassifier()
    rfc.fit(feat, targ)
    pred = rfc.predict([feat[0]])
    pred2 = rfc.predict(feat[0:5])

    assert pred is not None
    assert pred2 is not None

    # test model save and load
    with tempfile.TemporaryDirectory() as tmpdirname:
        file = Path(tmpdirname) / "rfc"
        rfc.save(file)
        assert file.exists()

        loaded = classifiers.MLEMRandomForestClassifier.load(file)

        assert loaded is not None
        assert isinstance(loaded.model, type(rfc.model))

        preds = rfc.predict(feat)
        loaded_preds = loaded.predict(feat)

        assert len(preds) == len(loaded_preds)

        comp = [all(a == b) for (a, b) in zip(preds, loaded_preds)]

        assert all(comp)
