from email.policy import default
import click
from click import UNPROCESSED
import pytest
from functools import partial
from strix.utilities.click_callbacks import framework_select, data_select, loss_select
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


    result = runner.invoke(cli_remeber_case, [], input="5")
    assert "[classification]:" in result.output.split("\n")[-2]
    assert "[selflearning]:" in result.output.split("\n")[-3]
    assert "[multitask]" in result.output.split("\n")[-4]

    result = runner.invoke(cli_normal_case, [], input="5")
    # print(result.output.split("\n"))
    assert "[1]:" in result.output.split("\n")[-2]
    assert "[2]:" in result.output.split("\n")[-3]
    assert "[classification]: 5" in result.output.split("\n")[-4]


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

@pytest.mark.parametrize('framework, default_loss', [("segmentation", "DCE"), ("multitask", ["DCE", "DCE"])])
@pytest.mark.parametrize('prompt', [True, False])
def test_click_loss_option(framework, prompt, default_loss, runner, tmp_path):
    @command(
        context_settings={
            "default_map": {'framework': framework, 'criterion': default_loss, 'subtask1': 'segmentation', 'subtask2': 'classification'},
            "prompt_in_default_map": prompt
        }
    )
    @option("--subtask1", type=str, default='segmentation')
    @option("--subtask2", type=str, default='segmentation')
    @option("--framework", type=str, default="segmentation")
    @option("-L", "--criterion", type=UNPROCESSED, callback=loss_select, default=None)
    def cli_loss_case(subtask1, subtask2, framework, criterion):
        pass

    result = runner.invoke(cli_loss_case, [], input="1")
    print(f"Loss results ({framework}-{prompt}-{default_loss}):", result.output)

    if framework == "segmentation":
        if prompt:
            assert "Loss_fn" in result.output
        else:
            assert len(result.output) == 0
    elif framework == "multitask":
        if prompt:
            assert len(result.output) > 0 and '\n' in result.output
        else:
            assert len(result.output) == 0


        

