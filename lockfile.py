import os
import random

class Lockfile:

    def __init__(self, lock_file_path):
        self.lock_file_path = lock_file_path
        self.id = "%d-%d-%d" % tuple([int(random.random() * 100000000) for _ in range(3)])
        self.my_lock_file_path = self.lock_file_path + self.id

    def __enter__(self):
        while True:
            try:
                os.rename(self.lock_file_path, self.my_lock_file_path)
            except FileNotFoundError as e:
                continue
            try:
                f = open(self.my_lock_file_path)
            except FileNotFoundError as e:
                continue
            f.close()
            return

    def __exit__(self):
        os.rename(self.my_lock_file_path, self.lock_file_path)