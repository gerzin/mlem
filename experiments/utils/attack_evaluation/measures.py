import pandas as pd
from sklearn.metrics import classification_report

def compute_fidelity_and_reports(datasets, classifier1, classifier2):
    """
    Parameters:
        datasets - List of tuples (dataset_name, X, y)
    """

    output_dict = {}

    for name, X, y in datasets:
        labels1 = classifier1[1].predict(X)
        labels2 = classifier2[1].predict(X)

        # compute fidelity between the two classifiers on name
        fidelity = str(pd.DataFrame(labels1 == labels2).value_counts(normalize=True))
        report_1 = classification_report(y, labels1)
        report_2 = classification_report(y, labels2)

        # add fidelity on both classes
        output_dict[name] = {
            classifier1[0] : report_1,
            classifier2[0]: report_2,
            "fidelity": fidelity,
        }
        
    return output_dict

def save_fidelity_and_reports(filename, dictstat):
    with open(str(filename), "w") as f:
        for key in dictstat:
            stats = dictstat[key]
            f.write(f"{key=}\n")
            for key2 in stats:
                f.write(f"{key2}:\n")
                f.write(f"{stats[key2]}\n\n")
            f.write("*"*15+"\n")