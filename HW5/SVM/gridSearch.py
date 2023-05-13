from libsvm.svmutil import svm_train
import itertools


def gridSearch(X_train, y_train, **param):
    kernel_type = param.get("kernel_type", 0)
    C = param.get("C", [1])
    gamma = param.get("gamma", [1 / X_train.shape[1]])
    coef0 = param.get("coef0", [0])
    degree = param.get("degree", [3])

    combinations = [C, gamma, coef0, degree]
    best_acc = 0
    best_comb = None
    for comb in list(itertools.product(*combinations)):
        acc = svm_train(y_train, X_train, f"-q -t {kernel_type} -v 3 -c {comb[0]} -g {comb[1]} -r {comb[2]} -d {comb[3]}")
        if acc > best_acc:
            best_acc = acc
            best_comb = comb

    print(f"best combination (C, gamma, coef0, degree): {best_comb}\tbest accuracy: {best_acc}")
    return best_comb, best_acc