import argparse
import numpy as np
from algo import LSE, Newton, predict, estimateError
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./testfile.txt', help='file path')
    parser.add_argument('--n', default=2, type=int, help='number of polynomial bases')
    parser.add_argument('--lamb', default=0, type=int, help='coefficient of regularization term (only for LSE)')

    args = parser.parse_args()
    return args

def read_file(path):
    x = []
    y = []
    with open(path) as f:
        for line in f.readlines():
            s = line.split(',')
            x.append(float(s[0]))
            y.append(float(s[1]))
    return x, y

def create_matrix(x, y, n):
    A = []
    for element in x:
        r = []
        for i in range(n):
            r.append(element ** i)
        A.append(r)
    
    A = np.array(A)
    b = np.array(y).reshape(-1, 1)

    return A, b

def print_result(x, pred, n):
    print("Fitting line: ", end='')
    for i in range(n-1, -1, -1):
        if i != 0:
            print(f"{x.item(i): .12f}X^{i} ", end='')
            if x[i-1] > 0:
                print("+", end='') 
        else:
            print(f"{x.item(i): .12f}")
    
    print(f"Total error: {estimateError(pred, b): .12f}\n")

def visualization(x0, y,  pred_lse, pred_newton):
    fig, axes = plt.subplots(2)

    axes[0].plot(x0, y, 'ro')
    axes[0].plot(x0, pred_lse, 'k-')

    axes[1].plot(x0, y, 'ro')
    axes[1].plot(x0, pred_newton, 'k-')

    plt.savefig('visualization.png') 

if __name__ == '__main__':
    args = parse_args()
    x0, y = read_file(args.path)
    A, b = create_matrix(x0, y, args.n)

    print("LSE:")
    x_lse = LSE(A, b, args.lamb)
    pred_lse = predict(A, x_lse)
    print_result(x_lse, pred_lse, args.n)

    print("Newton's method:")
    x_newton = Newton(A, b)
    pred_newton = predict(A, x_newton)
    print_result(x_newton, pred_newton, args.n)
    
    visualization(x0, y, pred_lse, pred_newton)