import click
import pytest
from functools import partial
from strix.utilities.click_callbacks import NumericChoice as Choice, dump_params
from strix.utilities.click import OptionEx, CommandEx
from strix.utilities.utils import get_items

option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)

@pytest.mark.parametrize('value', [True, False])
def test_prompt_cond(value, runner):
    @command()
    @option("-d", type=bool, default=value)
    @option("-g", prompt=True,  type=str, default='week', prompt_cond=lambda ctx: ctx.params['d'])
    def cli_prmopt_case(d, g):
        pass
    
    result = runner.invoke(cli_prmopt_case, [], input=None)
    if value:
        assert "[week]" in result.output
    else:
        assert len(result.output) == 0

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