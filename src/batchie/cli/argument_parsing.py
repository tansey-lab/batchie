import argparse
import numpy.random


class KVAppendAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary.
    """

    def __call__(self, parser, args, values, option_string=None):
        assert len(values) == 1
        try:
            (k, v) = values[0].split("=", 2)
        except ValueError as ex:
            raise argparse.ArgumentError(
                self, f'could not parse argument "{values[0]}" as k=v format'
            )
        d = getattr(args, self.dest) or {}
        d[k] = v
        setattr(args, self.dest, d)


def str_to_bool(s):
    """
    Convert a string to a boolean.
    :param s: A string.
    :return: A boolean.
    """
    if s.lower() in ["true", "t", "yes", "y", "1"]:
        return True
    elif s.lower() in ["false", "f", "no", "n", "0"]:
        return False
    else:
        raise ValueError(f"Could not convert '{s}' to a boolean.")


def cast_dict_to_type(k_v_string: dict[str, str], k_v_types: dict[str, type]):
    """
    Cast a dictionary of strings to a dictionary of types.
    :param k_v_string: A dictionary of strings.
    :param k_v_types: A dictionary of types.
    :return: A dictionary of types.
    """

    converters = {
        bool: str_to_bool,
        int: int,
        float: float,
        str: str,
    }

    return {
        k: converters.get(k_v_types[k], k_v_types[k])(v) for k, v in k_v_string.items()
    }


def get_prng_from_seed_argument(args):
    better_seed = numpy.random.SeedSequence(args.seed).generate_state(1)[0]
    return numpy.random.default_rng(better_seed)
