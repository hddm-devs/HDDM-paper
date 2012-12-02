import os
import glob

if __name__ == "__main__":
    filenames = glob.glob('temp/*.dat')
    for file in filenames:
        if os.stat(file).st_size == 257:
            os.remove(file)