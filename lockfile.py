import os
import random
import time

class Lockfile:

    def __init__(self, lock_file_path, verbose=False):
        self.verbose = verbose
        self.lock_file_path = lock_file_path
        self.id = "%d-%d-%d" % tuple([int(random.random() * 100000000) for _ in range(3)])
        self.my_lock_file_path = self.lock_file_path + '.' + self.id

    def __enter__(self):
        while True:
            try:
                os.rename(self.lock_file_path, self.my_lock_file_path)
                f = open(self.my_lock_file_path)
                f.close()
                return
            except FileNotFoundError as e:
                if self.verbose:
                    print('Lockfile %s is locked, retrying..' % self.lock_file_path)
                time.sleep(0.1)
                continue

    def __exit__(self, *args, **kwargs):
        os.rename(self.my_lock_file_path, self.lock_file_path)