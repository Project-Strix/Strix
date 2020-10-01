import os
import click

def split_input_str(value):
    return [ float(s) for s in value.split(',')] if value is not None else None

def _prompt(prompt_str, data_type, default_value):
    return click.prompt('\tInput {} for lr strategy'.format(prompt_str),\
                            type=data_type, default=default_value, value_proc=split_input_str)

if __name__ == "__main__":
    a = _prompt('TTTEST', tuple, (0.1,4.9))
    print(a)