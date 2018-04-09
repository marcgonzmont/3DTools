from os.path import isfile, join, altsep, isdir, basename
from os import listdir, makedirs, errno
from natsort import natsorted, ns
from numba import jit
from fnmatch import fnmatch
from matplotlib import pyplot as plt


def getSamples(path, ext= ''):
    '''
    Auxiliary function that extracts file names from a given path based on extension
    :param path: source path
    :param ext: file extension
    :return: array with samples
    '''
    samples = [altsep.join((path, f)) for f in listdir(path)
              if isfile(join(path, f)) and f.endswith(ext)]

    if len(samples) == 0:
        print("ERROR!!! ARRAY OF SAMPLES IS EMPTY (check file extension)")

    return samples


def splitSamples(all_images, tmp_substr, usr_substr):
    images = []
    templates = []

    for img in all_images:
        if fnmatch(img, tmp_substr):
            templates.append(img)
        elif fnmatch(img, usr_substr):
            images.append(img)

    images = natSort(images)
    templates = natSort(templates)
    return images, templates


def makeDir(path):
    '''
    To create output path if doesn't exist
    see: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
    :param path: path to be created
    :return: none
    '''
    try:
        makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

@jit
def natSort(list):
    '''
    Sort frames with human method
    see: https://pypi.python.org/pypi/natsort
    :param list: list that will be sorted
    :return: sorted list
    '''
    return natsorted(list, alg=ns.IGNORECASE)

def plotImages(titles, images, title, row, col):
    fig = plt.figure()
    for i in range(len(images)):
        plt.subplot(row, col, i + 1), plt.imshow(images[i])
        if len(titles) != 0:
            plt.title(titles[i])
        plt.gray()
        plt.axis('off')
    fig.suptitle(title, fontsize=14)
    plt.show()