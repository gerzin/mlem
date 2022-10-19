
class SKLearnAdapter:
    def __init__(self, classifier) -> None:
        self.classifier = classifier
    def __call__(self, x):
        return self.classifier.predict(x)