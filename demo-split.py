import os
import random


def get_dirs(path: str):
    #
    allpath = []

    for filepath, dirnames, filenames in os.walk(path):

        for dirname in dirnames:
            #
            if 's' in dirname:
                #
                allpath.append(filepath.split('\\')[-2] + '/' + filepath.split('\\')[-1] + '/' + dirname)

    # random.randint

    trains = []
    tests = []

    for path in allpath:

        if random.random() > 0.9:
            tests.append(path)
        else:
            trains.append(path)

    with open('train.txt', 'w') as f:

        for train in trains:
            #
            f.write(train + '\n')

    with open('val.txt', 'w') as f:

        for test in tests:
            #
            f.write(test + '\n')


if __name__ == '__main__':
    #
    get_dirs('F:\datasets\hq3_preprocessed\hq')
