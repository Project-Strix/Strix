import click
from click import UNPROCESSED
from functools import partial
import pytest
from strix.utilities.click import OptionEx, CommandEx
from strix.utilities.click_callbacks import loss_select

option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)

@pytest.mark.parametrize('value', ["1", "2"])
def test_multitask_loss(value, runner, tmp_path):
    @command()
    @option("--framework", type=str, default="multitask")
    @option("--criterion", type=UNPROCESSED, callback=loss_select, default=None, help="loss criterion type")
    def cli_test_case(framework, criterion):
        print("criterion:", criterion)

    result = runner.invoke(cli_test_case, [], input=value)
    if value == "1":
        assert "Input Weight" not in result.output
    elif value == "2":
        assert "Input Weight" in result.output


