import click
import pytest
from functools import partial
from strix.utilities.click_callbacks import NumericChoice as Choice, freeze_option
from strix.utilities.click import OptionEx, CommandEx

option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)

@pytest.mark.parametrize('value', [True, False])
def test_prompt_cond(value, runner):
    @command()
    @option("-d", is_flag=True, default=value)
    @option("-p", prompt=True,  type=str, default='week', prompt_cond=lambda ctx: ctx.params['d'])
    def cli_prmopt_case2(d, p):
        pass

    result = runner.invoke(cli_prmopt_case2, ["-d"], input=None)
    if value:
        assert len(result.output) == 0
    else:
        assert "[week]" in result.output

@pytest.mark.parametrize('value', [True, False])
def test_prompt_cond_remeber(value, runner):
    @command(context_settings={
        "prompt_in_default_map": True, "default_map": {"p": "year"}
    })
    @option("-p", prompt=True, type=str, default='week', prompt_cond=lambda x: value)
    def cli_prompt_case(p):
        pass

    result = runner.invoke(cli_prompt_case, [], input=None)
    print('->', result.output)
    if value:
        assert "[year]" in result.output
    else:
        assert len(result.output) == 0

