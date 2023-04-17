import numpy as np


def b2i(b, signed=True):
    return int.from_bytes(bytes=b, byteorder="big", signed=signed)


def load_image(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        magic = b2i(data[:4])

        if magic != 2051:
            raise IOError("File magic number error")

        num = b2i(data[4:8])
        row = b2i(data[8:12])
        col = b2i(data[12:16])
        image = np.frombuffer(data, dtype=np.uint8, offset=16)
    return image.reshape(num, row, col)


def load_label(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        magic = b2i(data[:4])

        if magic != 2049:
            raise IOError("File magic number error")

        num = b2i(data[4:8])
        label = np.frombuffer(data, dtype=np.uint8, offset=8)
    return label.reshape(num)