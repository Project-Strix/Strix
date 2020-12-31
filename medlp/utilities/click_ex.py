import inspect

from termcolor import colored
from click import Choice, ParamType, Option, confirm, Command
from click.types import convert_type, Path
from click.utils import echo
from click.exceptions import Abort, UsageError
from click.termui import visible_prompt_func, hidden_prompt_func

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

def prompt_ex(text, default=None, hide_input=False, confirmation_prompt=False,
              type=None, value_proc=None, prompt_suffix=': ', show_default=True,
              err=False, show_choices=True, color=None):
    result = None

    def prompt_func(text):
        f = hide_input and hidden_prompt_func or visible_prompt_func
        try:
            # Write the prompt separately so that we get nice
            # coloring through colorama on Windows
            echo(text, nl=False, err=err)
            return f('')
        except (KeyboardInterrupt, EOFError):
            # getpass doesn't print a newline if the user aborts input with ^C.
            # Allegedly this behavior is inherited from getpass(3).
            # A doc bug has been filed at https://bugs.python.org/issue24711
            if hide_input:
                echo(None, err=err)
            raise Abort()

    if value_proc is None:
        value_proc = convert_type(type, default)

    prompt = _build_prompt(text, prompt_suffix, show_default, default, show_choices, type)
    if color is not None:
        prompt = colored(prompt, color=color)

    while 1:
        while 1:
            value = prompt_func(prompt)
            if value:
                break
            elif default is not None:
                if isinstance(value_proc, Path):
                    # validate Path default value(exists, dir_okay etc.)
                    value = default
                    break
                return default
        try:
            result = value_proc(value)
        except UsageError as e:
            echo('Error: %s' % e.message, err=err)
            continue
        if not confirmation_prompt:
            return result
        while 1:
            value2 = prompt_func('Repeat for confirmation: ')
            if value2:
                break
        if value == value2:
            return result
        echo('Error: the two entered values do not match', err=err)

class OptionEx(Option):
    def __init__(self, param_decls=None, show_default=False,
                    prompt=False, confirmation_prompt=False,
                    hide_input=False, is_flag=None, flag_value=None,
                    multiple=False, count=False, allow_from_autoenv=True,
                    type=None, help=None, hidden=False, show_choices=True,
                    show_envvar=False, **attrs):
        super(OptionEx, self).__init__(
            param_decls=param_decls,
            show_default=show_default,
            prompt=prompt,
            confirmation_prompt=confirmation_prompt,
            hide_input=hide_input,
            is_flag=None,
            flag_value=flag_value,
            multiple=multiple,
            count=count,
            allow_from_autoenv=allow_from_autoenv,
            type=type,
            help=help,
            hidden=hidden,
            show_choices=show_choices,
            show_envvar=show_envvar,
            **attrs
        )
    
    def prompt_for_value(self, ctx):
        """This is an alternative flow that can be activated in the full
        value processing if a value does not exist.  It will prompt the
        user until a valid value exists and then returns the processed
        value as result.
        """
        # Calculate the default before prompting anything to be stable.
        default = self.get_default(ctx)

        # If this is a prompt for a flag we need to handle this
        # differently.
        if self.is_bool_flag:
            return confirm(self.prompt, default)

        return prompt_ex(self.prompt, default=default, type=self.type,
                         hide_input=self.hide_input, show_choices=self.show_choices,
                         confirmation_prompt=self.confirmation_prompt,
                         value_proc=lambda x: self.process_value(ctx, x))

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

def optionex(*param_decls, **attrs):
    """Attaches an option to the command.  All positional arguments are
    passed as parameter declarations to :class:`Option`; all keyword
    arguments are forwarded unchanged (except ``cls``).
    This is equivalent to creating an :class:`Option` instance manually
    and attaching it to the :attr:`Command.params` list.

    :param cls: the option class to instantiate.  This defaults to
                :class:`Option`.
    """
    def _param_memo(f, param):
        if isinstance(f, Command):
            f.params.append(param)
        else:
            if not hasattr(f, '__click_params__'):
                f.__click_params__ = []
            f.__click_params__.append(param)

    def decorator(f):
        # Issue 926, copy attrs, so pre-defined options can re-use the same cls=
        option_attrs = attrs.copy()

        if 'help' in option_attrs:
            option_attrs['help'] = inspect.cleandoc(option_attrs['help'])
        OptionClass = option_attrs.pop('cls', OptionEx)
        _param_memo(f, OptionClass(param_decls, **option_attrs))
        return f
    return decorator
