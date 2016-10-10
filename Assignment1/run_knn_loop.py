"""
Anita Bahmanyar
Student #: 998909098
"""
from   utils            import *
from   run_knn          import *
import matplotlib.pylab as     plt
import sys

set_type = sys.argv[1]
color    = sys.argv[2]

train_inputs, train_targets = load_train()
valid_inputs, valid_targets = load_valid()
test_inputs , test_targets  = load_test()

kvals      = [1,3,5,7,9]
frac_array = np.zeros(len(kvals))

for i in range(len(kvals)):
    
    if set_type == "valid": 
        valid_labels = run_knn(kvals[i], train_inputs, train_targets, valid_inputs)

    if set_type == "test":
        valid_labels = run_knn(kvals[i], train_inputs, train_targets, test_inputs)
        
    correct = 0
    for j in range(len(valid_labels)):
        if set_type == "valid":
            if valid_labels[j][0] == valid_targets[j][0]:
                correct += 1

        if set_type == "test":
            if valid_labels[j][0] == test_targets[j][0]:
                correct += 1
                
    frac = float(correct)/len(valid_labels)
    frac_array[i] = frac


plt.ion()
plt.plot(kvals, frac_array, color=color, linewidth=2, label=set_type)
plt.plot(kvals, frac_array, "o", color=color, label="")
plt.xlabel("k", fontsize=20)
plt.ylabel("classification rate", fontsize=15)
plt.title("Classification rate as a function of k")
plt.xlim(0.9,9.1)
plt.ylim(0.9,1)
plt.grid(True)
plt.legend(loc='best')

# computing the classification rate for test
