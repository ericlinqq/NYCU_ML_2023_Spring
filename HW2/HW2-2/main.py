import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./testfile.txt',
                        help='path of the data file')
    parser.add_argument('--a', type=int, default=0,
                        help='parameter a for the initial beta prior')
    parser.add_argument('--b', type=int, default=0,
                        help='parameter b for the initial beta prior')
    parser.add_argument('--record_dir', default='./',
                        help='path of the directory to store record file')
    args = parser.parse_args()
    return args


def load_data(path):
    with open(path, 'r') as f:
        cases = []
        for line in f:
            line = line.strip('\n')
            cases.append(line)

    return cases


def binomial(n, k, p):
    def combination(n, k):
        def factorial(n):
            f = []
            f.append(1)
            f.append(1)
            for i in range(2, n+1):
                f.append(f[i-1] * i)
            return f

        f = factorial(n)
        return f[n] / (f[k] * f[n-k])

    return combination(n, k) * (p ** k) * ((1. - p) ** (n - k))


def count_char(case, char):
    count = 0
    for c in case:
        if c == char:
            count += 1
    return count


def online_learning(a, b, cases, record_dir):
    filename = f"record_{a}_{b}.txt"
    path = os.path.join(record_dir, filename)
    with open(path, 'w') as f:
        for i, case in enumerate(cases):
            print(f"case {i+1}: {case}", file=f)
            N = len(case)
            head = count_char(case, '1')
            p = head / N
            likelihood = binomial(N, head, p)
            print(f"Likelihood: {likelihood}", file=f)
            print(f"Beta prior:     a = {a}  b = {b}", file=f)
            a += head
            b += N - head
            print(f"Beta posterior: a = {a}  b = {b}", file=f)
            print(file=f)


if __name__ == '__main__':
    args = parse_args()
    cases = load_data(args.data_path)
    online_learning(args.a, args.b, cases, args.record_dir)
