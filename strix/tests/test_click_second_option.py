import click
import pytest
from functools import partial
from strix.utilities.click_callbacks import framework_select, data_select
from strix.utilities.click_callbacks import NumericChoice as Choice
from strix.utilities.click import OptionEx, CommandEx
from strix.utilities.enum import FRAMEWORKS

option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)

def test_click_subtasks_option(runner, tmp_path):
    last_options = {"framework": "multitask", "subtask1": "selflearning", "subtask2": "classification"}

    @command(context_settings={"default_map": last_options, "prompt_in_default_map": True})
    @option("--framework", prompt=True, type=Choice(FRAMEWORKS), default="classification", callback=framework_select)
    def cli_remeber_case(framework):
        pass

    @command()
    @option("--framework", prompt=True, type=Choice(FRAMEWORKS), default="classification", callback=framework_select)
    def cli_normal_case(framework):
        pass


    result = runner.invoke(cli_remeber_case, [], input="6")
    assert "[classification]:" in result.output.split("\n")[-2]
    assert "[selflearning]:" in result.output.split("\n")[-3]
    assert "[multitask]" in result.output.split("\n")[-4]

    result = runner.invoke(cli_normal_case, [], input="6")
    # print(result.output.split("\n"))
    assert "[1]:" in result.output.split("\n")[-2]
    assert "[2]:" in result.output.split("\n")[-3]
    assert "[classification]: 6" in result.output.split("\n")[-4]


def test_click_datalist_option(runner, tmp_path):
    @command()
    @option("--tensor-dim", type=str, default='2D')
    @option("--framework", type=str, default="segmentation")
    @option("--datalist", type=str, callback=data_select, default=None)
    def cli_datalist_case(tensor_dim, framework, datalist):
        pass

    result = runner.invoke(cli_datalist_case, [], input="1\n")
    assert len(result.output) > 1


    last_options = {"datalist": "SyntheticData"}
    @command(context_settings={"default_map": last_options, "prompt_in_default_map": True})
    @option("--tensor-dim", type=str, default='2D')
    @option("--framework", type=str, default="segmentation")
    @option("--datalist", type=str, callback=data_select, default=None)
    def cli_datalist_remeber_case(tensor_dim, framework, datalist):
        pass
    
    result = runner.invoke(cli_datalist_remeber_case, [], input="1\n")
    print("Result output1:", result.output)
    assert len(result.output) > 1

    last_options = {"datalist": "SyntheticData"}
    @command(context_settings={"default_map": last_options})
    @option("--tensor-dim", type=str, default='2D')
    @option("--framework", type=str, default="segmentation")
    @option("--datalist", type=str, callback=data_select, default=None)
    def cli_datalist_noremeber_case(tensor_dim, framework, datalist):
        pass

    result = runner.invoke(cli_datalist_noremeber_case, [], input="1\n")
    print("Result output2:", result.output)
    assert len(result.output) == 0