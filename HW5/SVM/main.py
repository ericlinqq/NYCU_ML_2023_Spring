import argparse
import os
from dataloader import load_data
from libsvm.svmutil import svm_train, svm_predict, svm_problem
from gridSearch import gridSearch
from precomputed_kernel import linearRBF


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/', type=str, help="input data path")
    parser.add_argument('--result_path', default='./result/', type=str, help="result path")
    parser.add_argument('--result_filename', default='output.txt', type=str, help="output result filename")
    args = parser.parse_args()

    return args


if __name__ == '__main__':    
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise "input data path does not exist !!!!"
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    X_train, y_train, X_test, y_test = load_data(args.data_path)

    result_file = os.path.join(args.result_path, args.result_filename)
    f = open(result_file, 'w')
    print(f"output is saved in {result_file}.")

    kernel_type = {"linear": 0,
                   "polynomial": 1,
                   "RBF": 2,
                   "precomputed": 4}

# Part 1.
    print("Part 1.", file=f)
    for key, value in kernel_type.items():
        if key == 'precomputed':
            continue
        m = svm_train(y_train, X_train, f"-q -t {value}")
        p_labels, p_acc, p_vals = svm_predict(y_test, X_test, m, "-q")
        print(f"kernel_type: {key}\ttesting accuracy: {p_acc[0]:.2f}", file=f)
    print("-" * 60, file=f)

# Part 2.
    print("Part 2.", file=f)
    ## Linear kernel
    print("linear kernel", file=f)
    param = {"kernel_type": kernel_type['linear'], "C": [10**x for x in range(-5, 6)]}
    best_comb, best_acc = gridSearch(X_train, y_train, **param)
    print(f"best combination (C): {(best_comb[0])}\tbest training accuracy: {best_acc:.2f}", file=f)
    m = svm_train(y_train, X_train, f"-q -t {kernel_type['linear']} -c {best_comb[0]}")
    p_labels, p_acc, p_vals = svm_predict(y_test, X_test, m, "-q")
    print(f"after grid search testing accuracy: {p_acc[0]:.2f}", file=f)
    print(file=f)

    ## Polynomial kernel
    print("polynomail kernel", file=f)
    param = {"kernel_type": kernel_type['polynomial'], 
             "C": [10**x for x in range(-3, 4)],
             "gamma": [10**x for x in range(-3, 4)],
             "coef0": [x for x in range(-1, 2)],
             "degree": [x for x in range(2, 5)]}
    best_comb, best_acc = gridSearch(X_train, y_train, **param)
    print(f"best combination (C, gamma, coef0, degree): {best_comb}\tbest training accuracy: {best_acc:.2f}", file=f)
    m = svm_train(y_train, X_train, f"-q -t {kernel_type['polynomial']} -c {best_comb[0]} -g {best_comb[1]} -r {best_comb[2]} -d {best_comb[3]}")
    p_labels, p_acc, p_vals = svm_predict(y_test, X_test, m, "-q")
    print(f"after grid search testing accuracy: {p_acc[0]:.2f}", file=f)
    print(file=f)

    ## Radial basis kernel
    print("RBF kernel", file=f)
    param = {"kernel_type": kernel_type['RBF'], 
             "C": [10**x for x in range(-3, 4)],
             "gamma": [10**x for x in range(-3, 4)]}
    best_comb, best_acc = gridSearch(X_train, y_train, **param) 
    print(f"best combination (C, gamma): {(best_comb[0], best_comb[1])}\tbest training accuracy: {best_acc:.2f}", file=f)
    m = svm_train(y_train, X_train, f"-q -t {kernel_type['RBF']} -g {best_comb[1]}")
    p_labels, p_acc, p_vals = svm_predict(y_test, X_test, m, "-q")
    print(f"after grid search testing accuracy: {p_acc[0]:.2f}", file=f)
    print("-" * 60, file=f)

# Part 3.
    print("Part 3.", file=f)
    K = linearRBF(X_train, X_train, best_comb[1])
    KK = linearRBF(X_test, X_train, best_comb[1])
    prob = svm_problem(y_train, K, isKernel=True)
    m = svm_train(prob, f"-q -t {kernel_type['precomputed']} -c {best_comb[0]}")
    p_labels, p_acc, p_vals = svm_predict(y_test, KK, m, "-q")
    print(f"kernel_type: linear + RBF kernel\ttesting accuracy: {p_acc[0]:.2f}", file=f)
    print("-" * 60, file=f)