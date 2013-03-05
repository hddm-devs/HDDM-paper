import os
import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HDDM experiments.', add_help=True)
    parser.add_argument('folder', type=str)

    result = parser.parse_args()
    folder = result.folder

    filenames = glob.glob(os.path.join('simulations', folder, 'single_runs/*.dat*'))
    for file in filenames:
        if os.stat(file).st_size == 257:
            os.remove(file)