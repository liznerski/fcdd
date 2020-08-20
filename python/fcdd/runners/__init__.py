import os
import sys


class Tee(object):
    """
    Use withtin a 'with'-statement to duplicate everything printed on the stdout or stderr stream
    and write that duplicate in a file.
    :param path: path to the file where the duplicates are written to
    :param mode: which mode to use to write, e.g. 'a' to append (default)
    """
    def __init__(self, path: str, mode: str = 'a'):
        self.filename = path
        self.mode = mode
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = type(
            'ErrTee', (object, ), {
                'write': lambda data: self.err_write(data),
                'flush': lambda: self.err_flush()
            }
        )

    def __exit__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def err_write(self, data):
        self.file.write(data)
        self.stderr.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def err_flush(self):
        self.file.flush()
        self.stderr.flush()
