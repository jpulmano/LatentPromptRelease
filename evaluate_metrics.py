import os
import sys
from sklearn.metrics import f1_score, accuracy_score
from statistics import mean, stdev, median

import wandb

wandb.init(project="LatentPromptAnalysis-Metrics")

if len(sys.argv) < 4:
	print('use like: evaluate_metrics.py <path/to/time_vessels> <num_test> <predictions_file>')
	exit()

time_vessels_path = sys.argv[1]
num_tests = int(sys.argv[2])
predictions_file = sys.argv[3]

wandb.log({
	'time_vessels_path': time_vessels_path,
	'num_tests': num_tests
})

f1_positives = []
f1_negatives = []
accuracies = []

for test_num in range(num_tests):
	with open(os.path.join(time_vessels_path, 'time_vessels', str(test_num), predictions_file)) as test_file:
		lines = test_file.readlines()
		labels = [[int(p) for p in line.split(',')] for line in lines]


	labels_true = [label[0] for label in labels]
	labels_pred = [label[1] for label in labels]

	f1_positives.append(f1_score(y_true=labels_true, y_pred=labels_pred))
	f1_negatives.append(f1_score(y_true=labels_true, y_pred=labels_pred, pos_label=0))
	accuracies.append(accuracy_score(y_true=labels_true, y_pred=labels_pred))


print('f1_positives: ', f1_positives)
print('f1_negatives: ', f1_negatives)
print('accuracies: ', accuracies)

if len(f1_positives) > 1:
	print('F1 pos: {} ({}) : {}'.format(mean(f1_positives), stdev(f1_positives), median(f1_positives)))
	print('F1 neg: {} ({}) : {}'.format(mean(f1_negatives), stdev(f1_negatives), median(f1_negatives)))
	print('Accuracy: {} ({}) : {}'.format(mean(accuracies), stdev(accuracies), median(accuracies)))
	
	wandb.log({
		'F1_pos_mean': mean(f1_positives),
		'F1_pos_stddev': stdev(f1_positives),
		'F1_pos_median': median(f1_positives),
		'F1_neg_mean': mean(f1_negatives),
		'F1_neg_stddev': stdev(f1_negatives),
		'F1_neg_median': median(ff1_negatives),
		'Acc_mean': mean(accuracies),
		'Acc_stddev': stdev(accuracies),
		'Acc_median': median(accuracies),
	})




