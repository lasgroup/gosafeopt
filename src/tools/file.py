import os
import os.path
import imageio

from gosafeopt import aquisitions


def clearFiles(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file[0] != ".":
                os.remove(os.path.join(root, file))


def makeGIF():
    frames = []
    for _, _, files in os.walk("data/plot"):
        files.sort()
        for file in files:
            if file.endswith(".png"):
                image = imageio.imread("data/plot/{}".format(file))
                frames.append(image)
    imageio.mimsave("data/animation.gif", frames, format="GIF", duration=0.3)

def createFolderINE(PATH):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
