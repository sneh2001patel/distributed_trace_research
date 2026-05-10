try:
    from model import GraphClassifier
except ModuleNotFoundError:
    from classifier.model import GraphClassifier


TTGraphClassifier = GraphClassifier

