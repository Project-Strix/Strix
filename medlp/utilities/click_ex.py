from click import Choice, ParamType
from click.types import convert_type


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
            value = list(filter(lambda x : x is not ' ', value))
        elif value is None or value == '':
            return None

        types = (self.type,) * len(value)
        return tuple(ty(x, param, ctx) for ty, x in zip(types, value))


def _build_prompt(text, suffix, show_default=False, default=None, show_choices=True, type=None):
    prompt = text
    if type is not None and show_choices and isinstance(type, Choice):
        prompt += ' (' + ", ".join(map(str, type.choices.values())) + ')' if not type.show_index else \
                  ' (' + ", ".join(['%s:%s'%(idx,ch) for idx,ch in type.choices.items()]) + ')' 
    if default is not None and show_default:
        prompt = '%s [%s]' % (prompt, default)
    return prompt + suffix


class ChoiceEx(Choice):
    """The choice type allows a value to be checked against a fixed set
    of supported values. All of these values have to be strings.

    You should only pass a list or tuple of choices. Other iterables
    (like generators) may lead to surprising results.

    See :ref:`choice-opts` for an example.

    :param case_sensitive: Set to false to make choices case
        insensitive. Defaults to true.
    """

    name = 'choice'

    def __init__(self, choices, case_sensitive=True, show_index=False):
        super(ChoiceEx, self).__init__(
            choices={i:choice for i, choice in enumerate(choices)},
            case_sensitive=case_sensitive,
        )
        self.show_index = show_index

    def get_metavar(self, param):
        return '[%s]' % '|'.join(self.choices.values())

    def get_missing_message(self, param):
        if self.show_index:
            return 'Choose from:\n\t%s.' % ',\n\t'.join(['%s.%s'%(idx,ch) for idx,ch in self.choices.items()])
        else:
            return 'Choose from:\n\t%s.' % ',\n\t'.join(self.choices.values())

    def convert(self, value, param, ctx):
        # Exact match
        if self.show_index:
            try:
                idx = int(value)
            except:
                if value in self.choices.values():
                    return value
                self.fail('invaid index choice: %s. Please input integer index or correct value!' % (value))

            if idx in list(self.choices.keys()):
                return self.choices[idx]
            
            self.fail('invalid choice: %s. (choose from %s)' %
                      (value, ', '.join(map(str, self.choices.keys()))), param, ctx)
        else:
            if value in self.choices.values():
                return value

            # Match through normalization and case sensitivity
            # first do token_normalize_func, then lowercase
            # preserve original `value` to produce an accurate message in
            # `self.fail`
            normed_value = value
            normed_choices = self.choices.values()

            if ctx is not None and \
            ctx.token_normalize_func is not None:
                normed_value = ctx.token_normalize_func(value)
                normed_choices = [ctx.token_normalize_func(choice) for choice in
                                  self.choices.values()]

            if not self.case_sensitive:
                normed_value = normed_value.lower()
                normed_choices = [choice.lower() for choice in normed_choices]

            if normed_value in normed_choices:
                return normed_value

            self.fail('invalid choice: %s. (choose from %s)' %
                      (value, ', '.join(self.choices.values())), param, ctx)

    def __repr__(self):
        if not self.show_index:
            return 'Choice(%r)' % list(self.choices.values())
        else:
            return 'Choice(%r)' % ['{}.{}'.format(idx,ch) for idx, ch in self.choices.items()]


class NumericChoice(Choice):
    def __init__(self, choices, **kwargs):
        self.choicemap = {}
        choicestrs = []
        for i, choice in enumerate(choices, start=1):
            self.choicemap[i] = choice
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


def get_unknown_options(ctx):
    auxilary_params = {
        (ctx.args[i][2:] if str(ctx.args[i]).startswith("--") else ctx.args[i][1:]): ctx.args[i + 1]
        for i in range(0, len(ctx.args), 2)
    }
    return auxilary_params
