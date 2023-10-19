import argparse


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


def cast_dict_to_type(k_v_string: dict[str, str], k_v_types: dict[str, type]):
    """
    Cast a dictionary of strings to a dictionary of types.
    :param k_v_string: A dictionary of strings.
    :param k_v_types: A dictionary of types.
    :return: A dictionary of types.
    """
    return {k: k_v_types[k](v) for k, v in k_v_string.items()}
