"""Utility functions.

Taken from https://github.com/nryant/dscore
"""
import itertools
import sys
import click
import ast


def error(msg, file=sys.stderr):
    """Log error message ``msg`` to stderr."""
    msg = "ERROR: %s" % msg
    print(msg, file=file)


def info(msg, print_level=False, file=sys.stdout):
    """Log info message ``msg`` to stdout."""
    if print_level:
        msg = "INFO: %s" % msg
    print(msg, file=file)


def warn(msg, file=sys.stderr):
    """Log warning message ``msg`` to stderr."""
    msg = "WARNING: %s" % msg
    print(msg, file=file)


def xor(x, y):
    """Return truth value of ``x`` XOR ``y``."""
    return bool(x) != bool(y)


def format_float(x, n_digits=3):
    """Format floating point number for output as string.

    Parameters
    ----------
    x : float
        Number.

    n_digits : int, optional
        Number of decimal digits to round to.
        (Default: 3)

    Returns
    -------
    s : str
        Formatted string.
    """
    fmt_str = "%%.%df" % n_digits
    return fmt_str % round(x, n_digits)


def clip(x, lower, upper):
    """Clip ``x`` to [``lower``, ``upper``]."""
    return min(max(x, lower), upper)


def groupby(iterable, keyfunc):
    """Wrapper around ``itertools.groupby`` which sorts data first."""
    iterable = sorted(iterable, key=keyfunc)
    for key, group in itertools.groupby(iterable, keyfunc):
        yield key, group


# If an option is selected, other options become required
def command_required_option(require_name, require_map):
    class CommandOptionRequiredClass(click.Command):
        def invoke(self, ctx):
            require = ctx.params[require_name]
            if require not in require_map:
                raise click.ClickException(
                    "Unexpected value for --'{}': {}".format(require_name, require)
                )
            if (
                require_map[require] is not None
                and ctx.params[require_map[require].lower()] is None
            ):
                raise click.ClickException(
                    "With {}={} must specify option --{}".format(
                        require_name, require, require_map[require]
                    )
                )
            super(CommandOptionRequiredClass, self).invoke(ctx)

    return CommandOptionRequiredClass


# Class to accept list of arguments as Click option
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            return None
