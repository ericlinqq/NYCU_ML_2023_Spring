import cv2


def load_data(img_path):
    img = cv2.imread(img_path)
    height, width, channel = img.shape
    return img.reshape(-1, channel), height, width