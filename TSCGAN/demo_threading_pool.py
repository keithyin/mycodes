from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def func(i):
    while True:
        i += 1


pool = multiprocessing.Pool()
ppe = ProcessPoolExecutor(max_workers=10)


