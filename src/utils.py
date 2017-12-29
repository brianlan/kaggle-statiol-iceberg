import os


def mkdir(d):
    try:
        os.mkdir(d)
    except FileExistsError:
        pass


def mkdir_r(d):
    if not os.path.exists(d):
        mkdir_r(os.path.dirname(d))

    mkdir(d)
