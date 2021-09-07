from click import Choice, ParamType
from click.types import convert_type
from utils_cw.utils import Print
from types import SimpleNamespace

class DynamicTuple(ParamType):

    def __init__(self, input_type):
        self.type = convert_type(input_type)

    @property
    def name(self):
        return "< Dynamic Tuple >"

    def convert(self, value, param, ctx):
        # Hotfix for prompt input
        if isinstance(value, str):
            if ',' in value:
                sep = ','
            elif ';' in value:
                sep = ';'
            else:
                sep = ' '

            value = value.strip().split(sep)
            value = list(filter(lambda x : x != ' ', value))
        elif value is None or value == '':
            return None

        types = (self.type,) * len(value)
        return tuple(ty(x, param, ctx) for ty, x in zip(types, value))


class NumericChoice(Choice):
    def __init__(self, choices, **kwargs):
        self.choicemap = {}
        choicestrs = []
        for i, choice in enumerate(choices, start=1):
            self.choicemap[i] = choice
            if len(choices) > 5:
                choicestrs.append(f"\n\t{i}: {choice}")
            else:
                choicestrs.append(f"{i}: {choice}")

        super().__init__(choicestrs, **kwargs)

    def convert(self, value, param, ctx):
        try:
            return self.choicemap[int(value)]
        except ValueError as e:
            if value in self.choicemap.values():
                return value
            self.fail(
                f'invaid index choice: {value}. Please input integer index or correct value!'
                f'Error msg: {e}'
            )
        except KeyError as e:
            self.fail(f'invalid choice: {value}. (choose from {self.choicemap})', param, ctx)


def _convert_type(var, types=[float, str]):
    for type_ in types: 
        try:
            return type_(var)
        except ValueError as e:
            pass
    return var


def get_unknown_options(ctx, verbose=False):
    auxilary_params = {}

    if isinstance(ctx, SimpleNamespace): #! temp solution
        return auxilary_params

    for i in range(0, len(ctx.args), 2):  #Todo: how to handle flag auxilary params?
        if str(ctx.args[i]).startswith("--"):
            auxilary_params[ctx.args[i][2:].replace('-', '_')] = _convert_type(ctx.args[i + 1])
        elif str(ctx.args[i]).startswith("-"):
            auxilary_params[ctx.args[i][1:].replace('-', '_')] = _convert_type(ctx.args[i + 1])
        else:
            Print("Got invalid argument:", ctx.args[i], color='y', verbose=verbose)

    Print("Got auxilary params:", auxilary_params, color='y', verbose=verbose)
    return auxilary_params
