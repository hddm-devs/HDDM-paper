import os
import glob
import estimate as est

if __name__ == "__main__":
    filenames = glob.glob(est.SINGLE_RUNS_FOLDER + '/*.dat*')
    for file in filenames:
        if os.stat(file).st_size == 257:
            os.remove(file)