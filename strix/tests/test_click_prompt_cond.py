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
    def cli_prmopt_case(p, g):
        pass
    
    result = runner.invoke(cli_prmopt_case, [], input=None)
    print("----\n", result.output)
    if value:
        assert "[week]" in result.output
    else:
        assert len(result.output) == 0
