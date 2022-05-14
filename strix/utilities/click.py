import typing as t
from click import Choice, ParamType, Option, Context, Parameter, Command
from click.core import ParameterSource
from click.parser import _flag_needs_value
from click.types import convert_type

###################### Extension of click ################################

class ContextEx(Context):
    def __init__(
        self,
        command: "Command",
        parent: t.Optional["Context"] = None,
        info_name: t.Optional[str] = None,
        obj: t.Optional[t.Any] = None,
        auto_envvar_prefix: t.Optional[str] = None,
        default_map: t.Optional[t.Dict[str, t.Any]] = None,
        terminal_width: t.Optional[int] = None,
        max_content_width: t.Optional[int] = None,
        resilient_parsing: bool = False,
        allow_extra_args: t.Optional[bool] = None,
        allow_interspersed_args: t.Optional[bool] = None,
        ignore_unknown_options: t.Optional[bool] = None,
        help_option_names: t.Optional[t.List[str]] = None,
        token_normalize_func: t.Optional[t.Callable[[str], str]] = None,
        color: t.Optional[bool] = None,
        show_default: t.Optional[bool] = None,
        prompt_in_default_map: bool = False,
    ) -> None:
        super().__init__(
            command,
            parent,
            info_name,
            obj,
            auto_envvar_prefix,
            default_map,
            terminal_width,
            max_content_width,
            resilient_parsing,
            allow_extra_args,
            allow_interspersed_args,
            ignore_unknown_options,
            help_option_names,
            token_normalize_func,
            color,
            show_default
        )
        self.prompt_in_default_map = prompt_in_default_map

class CommandEx(Command):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_class = ContextEx

class OptionEx(Option):
    def consume_value(self, ctx: Context, opts: t.Mapping[str, "Parameter"]) -> t.Tuple[t.Any, ParameterSource]:
        value, source = super().consume_value(ctx, opts)

        # The parser will emit a sentinel value if the option can be
        # given as a flag without a value. This is different from None
        # to distinguish from the flag not being given at all.
        prompt_in_default_map = ctx.prompt_in_default_map if hasattr(ctx, "prompt_in_default_map") else False

        if value is _flag_needs_value:
            if self.prompt is not None and not ctx.resilient_parsing:
                value = self.prompt_for_value(ctx)
                source = ParameterSource.PROMPT
            else:
                value = self.flag_value
                source = ParameterSource.COMMANDLINE

        elif (
            self.multiple
            and value is not None
            and any(v is _flag_needs_value for v in value)
        ):
            value = [self.flag_value if v is _flag_needs_value else v for v in value]
            source = ParameterSource.COMMANDLINE

        # The value wasn't set, or used the param's default, prompt if
        # prompting is enabled.
        elif (
            (
                source in {None, ParameterSource.DEFAULT} or \
                (prompt_in_default_map and source == ParameterSource.DEFAULT_MAP)
            )
            and self.prompt is not None
            and (self.required or self.prompt_required)
            and not ctx.resilient_parsing
        ):
            value = self.prompt_for_value(ctx)
            source = ParameterSource.PROMPT

        return value, source

class DynamicTuple(ParamType):
    def __init__(self, input_type):
        self.type = convert_type(input_type)

    @property
    def name(self):
        return "< Dynamic Tuple >"

    def convert(self, value, param, ctx):
        # Hotfix for prompt input
        if isinstance(value, str):
            if "," in value:
                sep = ","
            elif ";" in value:
                sep = ";"
            else:
                sep = " "

            value = value.strip().split(sep)
            value = list(filter(lambda x: x != " ", value))
        elif value is None or value == "":
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
                f"invaid index choice: {value}. Please input integer index or correct value!"
                f"Error msg: {e}"
            )
        except KeyError as e:
            self.fail(
                f"invalid choice: {value}. (choose from {self.choicemap})", param, ctx
            )

