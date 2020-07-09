import argparse
import re
from argparse import PARSER, REMAINDER, OPTIONAL, ZERO_OR_MORE, ArgumentError


def cast(val, typ, example):
    if typ in [list, tuple]:
        ls = val.replace('(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
        ls = [cast(l, type(example[0]) if example[0] is not None else str, example[0]) for l in ls]
        return ls
    elif typ in [bool]:
        if isinstance(val, str):
            try:
                number = bool(int(val))
            except ValueError:
                number = True
            return len(val) > 0 and val.lower() not in ['false'] and number
        else:
            return typ(val)
    else:
        return typ(val)


class StoreDictKeyValPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyValPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        keyvals = {}
        for kv in values:
            k, v = kv.split("=")
            for kind, dic in self.choices.items():
                if k in dic:
                    type_default_val = type(dic[k]) if dic[k] is not None else str
                    v = cast(v, type_default_val, dic[k])
                    break
            keyvals[k] = v
        setattr(namespace, self.dest, keyvals)


class ArgumentParser(argparse.ArgumentParser):
    def _get_values(self, action, arg_strings):
        # ----- changed -----
        if action.choices is not None and len(action.choices) > 0 and type(action.choices) not in [str, dict]:
            matches = []
            for arg_string in arg_strings:
                regex = re.compile(arg_string)
                matches.extend([choice for choice in action.choices if re.fullmatch(regex, choice)])
                if len([choice for choice in action.choices if re.fullmatch(regex, choice)]) < 1:
                    raise ValueError('No match for option {} with argument {}'.format(action.dest, arg_string))
            arg_strings = matches
        # ----- changed -----

        # for everything but PARSER, REMAINDER args, strip out first '--'
        if action.nargs not in [PARSER, REMAINDER]:
            try:
                arg_strings.remove('--')
            except ValueError:
                pass

        # optional argument produces a default when not present
        if not arg_strings and action.nargs == OPTIONAL:
            if action.option_strings:
                value = action.const
            else:
                value = action.default
            if isinstance(value, str):
                value = self._get_value(action, value)
                self._check_value(action, value)

        # when nargs='*' on a positional, if there were no command-line
        # args, use the default if it is anything other than None
        elif (not arg_strings and action.nargs == ZERO_OR_MORE and
              not action.option_strings):
            if action.default is not None:
                value = action.default
            else:
                value = arg_strings
            self._check_value(action, value)

        # single argument or optional argument produces a single value
        elif len(arg_strings) == 1 and action.nargs in [None, OPTIONAL]:
            arg_string, = arg_strings
            value = self._get_value(action, arg_string)
            self._check_value(action, value)

        # REMAINDER arguments convert all values, checking none
        elif action.nargs == REMAINDER:
            value = [self._get_value(action, v) for v in arg_strings]

        # PARSER arguments convert all values, but check only the first
        elif action.nargs == PARSER:
            value = [self._get_value(action, v) for v in arg_strings]
            self._check_value(action, value[0])

        # all other types of nargs produce a list
        else:
            value = [self._get_value(action, v) for v in arg_strings]
            for v in value:
                self._check_value(action, v)

        # return the converted value
        return value

    def _check_value(self, action, value):
        if type(action) == StoreDictKeyValPair:
            if type(action.choices) not in [dict]:
                raise ArgumentError(action, 'Choices for ActionType StoreDictKeyValPair must be a dict.')
            if not all([type(v) == dict for k, v in action.choices.items()]):
                raise ArgumentError(action, 'Choices for ActionType StoreDictKeyValPair must be a dict of dicts.')
            if len(value.split("=")) != 2:
                raise ArgumentError(
                    action, 'Argument of type StoreDictKeyValPair is not of style KEY=VAL for {}'.format(value)
                )
            all_keys = set(k for c in action.choices for k in list(action.choices[c].keys()))
            key, val = value.split("=")
            if key not in all_keys:
                raise ArgumentError(
                    action, 'Argument of type StoreDictKeyValPair has an unknown key {}'.format(key)
                )
            for kind, dic in action.choices.items():
                if key in dic:
                    type_default_val = type(dic[key]) if dic[key] is not None else str
                    try:
                        _ = cast(val, type_default_val, dic[key])
                    except ValueError:
                        raise ArgumentError(
                            action,
                            'Argument of type StoreDictKeyValPair cannot be cast to expected type, for {}'.format(value)
                        )
        else:
            super()._check_value(action, value)

    def rmv_argument(self, dest):
        actions = [a for a in self._actions if dest == a.dest]
        for a in actions:
            options = a.option_strings.copy()
            for option_string in options:
                self._handle_conflict_resolve(None, [(option_string, a)])
