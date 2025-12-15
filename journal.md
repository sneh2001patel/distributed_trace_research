## Re-weighting features

PodName has 28 unique values, and OpName has ~200 unique values this cause an inbalance when training the network, using indivual weights for each feature increases the accuracy for OP

Output: This straegy is not working need to do something else.
using weight class:
=== Reconstruction Evaluation ===
Total nodes evaluated: 106020
Pod accuracy: 64.30%
Op accuracy:  39.22%
Duration MAE: 16600.2427
Edge precision: 87.71%
Edge recall:    23.36%
Edge F1:        36.90%

without class:
=== Reconstruction Evaluation ===
Total nodes evaluated: 106020
Pod accuracy: 79.53%
Op accuracy:  52.45%
Duration MAE: 18702.8461
Edge precision: 70.19%
Edge recall:    32.25%
Edge F1:        44.20%


## Increasing Epochs to 200

Before I was training the model at 80 epochs, now I am training at 200 epochs I see a masive improvement in feature accruacy, recall, and F1 score. 

Before at 80 epochs:
=== Reconstruction Evaluation ===
Total nodes evaluated: 106020
Pod accuracy: 79.53%
Op accuracy:  52.45%
Duration MAE: 18702.8461
Edge precision: 70.19%
Edge recall:    32.25%
Edge F1:        44.20%

At 200 epochs:
=== Reconstruction Evaluation ===
Total nodes evaluated: 106020
Pod accuracy: 81.83%
Op accuracy:  62.47%
Duration MAE: 19535.0017
Edge precision: 82.18%
Edge recall:    46.27%
Edge F1:        59.20%

> TODO: Mess aruond with the recall and F1 precision, recall, and f1 score.



The weights did not help I am gettng a better results when not using the weights then when I do use them.
=== Reconstruction Evaluation ===
Total nodes evaluated: 106020
Pod accuracy: 64.30%
Op accuracy:  39.22%
Duration MAE: 16600.2427
Edge precision: 87.71%
Edge recall:    23.36%
Edge F1:        36.90%

without class:
=== Reconstruction Evaluation ===
Total nodes evaluated: 106020
Pod accuracy: 79.53%
Op accuracy:  52.45%
Duration MAE: 18702.8461
Edge precision: 70.19%
Edge recall:    32.25%
Edge F1:        44.20%

However I increased the number of epochs (not using the weights) and I got better results.
=== Reconstruction Evaluation ===
Total nodes evaluated: 106020
Pod accuracy: 81.83%
Op accuracy:  62.47%
Duration MAE: 19535.0017
Edge precision: 82.18%
Edge recall:    46.27%
Edge F1:        59.20%

Best results with the value:
when use_class_weights = False, op_loss_scale = 1.2, edge_threshold = 0.8

=== Reconstruction Evaluation ===
Total nodes evaluated: 106020
Pod accuracy: 81.49%
Op accuracy:  57.05%
Duration MAE: 19450406.2057
Edge precision: 77.94%
Edge recall:    52.54%
Edge F1:        62.77%

