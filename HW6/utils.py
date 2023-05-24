from imageio import mimsave
import os


def save_gif(img_list, gif_path="./gif", gif_name="result.gif", duration=2):
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    for img in img_list:
        img = img[:, :, ::-1]
    mimsave(os.path.join(gif_path, gif_name), img_list, 'GIF', duration=duration)
    return