import os
import sys


class Maintainer(object):
    def __init__(self, args, *restrictions):
        """
        Maintainer that maintains an args environment. It prohibits setting new paramters in
        maintained args, but allows changing existing ones. It is also possible to restrict
        further what and how existing ones can be changed.
        :param args: namespace -- some args that an argsparser yielded
        :param restrictions:
            [(str, iter OR any, func)] -- tuples mapping from key ([0]) to either range iter ([1])
            or comparision value any ([1]). During __setattr__ if a key is found in restrictions it is checked.
            The third component option is optional (defaults to identity) and defines some metric
            of key str that is applied first and which result is compared to be either:
                - in range iter
                - being equal to value any
            Also take care that iter is either list or tuple.
        """
        self.args = args
        object.__setattr__(self, '_restrictions', restrictions)

    def __getattribute__(self, item):
        if item == 'args':
            return object.__getattribute__(self, 'args')
        elif item == '_restrictions':
            return object.__getattribute__(self, '_restrictions')
        else:
            return getattr(object.__getattribute__(self, 'args'), item)

    def __setattr__(self, key, value):
        if key == 'args':
            try:
                _ = object.__getattribute__(self, 'args')
                object.__setattr__(self, key, value)
                print('WARNING: Maintainer received "args" attribute. Overwrote previous one.')
            except AttributeError:
                object.__setattr__(self, key, value)
        elif key in self.args:
            for tup in self._restrictions:
                if tup[0] == key:
                    check = tup[1]
                    metric = tup[2] if len(tup) > 2 else lambda x: x
                    if isinstance(check, tuple) or isinstance(check, list):
                        if metric(value) not in check:
                            raise ValueError(
                                'Maintainer forbids setting {} to {} because using metric {} that is not in {}'
                                .format(key, value, metric, check)
                            )
                    else:
                        if metric(value) != check:
                            raise ValueError(
                                'Maintainer forbids setting {} to {} because using metric {} that is not {}'
                                .format(key, value, metric, check)
                            )
            setattr(self.args, key, value)
        else:
            raise ValueError("{} not part of args. Maintainer forbids adding new arguments".format(key))

    def __str__(self):
        return str(self.args)


class Tee(object):
    """
    Use withtin a 'with'-statement to duplicate everything printed on the stdout or stderr stream
    and write that duplicate in a file
    :param path: path to the file where the duplicated are written to
    :param mode: which mode to use to write, e.g. 'a' to append (default)
    """
    def __init__(self, path, mode='a'):
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
