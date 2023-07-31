import os
import random


def get_dirs(path: str):
    #
    index = 1

    for filepath, dirnames, filenames in os.walk(path):

        for filename in filenames:
            #
            if 'mp4' in filename:
                #
                os.rename(filepath + '\\' + filename, filepath + '\\' + 's{:0>3d}.mp4'.format(index))

                index += 1


if __name__ == '__main__':
    #
    get_dirs('E:\\application\\BaiduNetdiskDownload\\新闻高清\\p1032')
