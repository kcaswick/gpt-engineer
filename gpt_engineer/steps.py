import inspect
import re
import subprocess

from enum import Enum
from typing import List, Union

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from termcolor import colored

from gpt_engineer.ai import AI
from gpt_engineer.chat_to_files import (
    format_file_to_input,
    get_code_strings,
    overwrite_files,
    to_files,
)
from gpt_engineer.db import DBs
from gpt_engineer.file_selector import ask_for_files
from gpt_engineer.learning import human_review_input

Message = Union[AIMessage, HumanMessage, SystemMessage]


def setup_sys_prompt(dbs: DBs) -> str:
    """
    Primes the AI with instructions as to how it should
    generate code and the philosophy to follow
    """
    return (
        dbs.preprompts["roadmap"]
        + dbs.preprompts["generate"]
        + "\nUseful to know:\n"
        + dbs.preprompts["philosophy"]
    )


def setup_sys_prompt_existing_code(dbs: DBs) -> str:
    """
    Similar to code generation, but using an existing code base.
    """
    return (
        dbs.preprompts["implement_on_existing"]
        + "\nUseful to know:\n"
        + dbs.preprompts["philosophy"]
    )


def get_prompt(dbs: DBs) -> str:
    """
    Loads the user's prompt for the project from prompt file
    (While we migrate we have this fallback getter)
    """
    assert (
        "prompt" in dbs.input or "main_prompt" in dbs.input
    ), "Please put your prompt in the file `prompt` in the project directory"

    if "prompt" not in dbs.input:
        print(
            colored("Please put the prompt in the file `prompt`, not `main_prompt", "red")
        )
        print()
        return dbs.input["main_prompt"]

    return dbs.input["prompt"]


def curr_fn() -> str:
    """
    Get the name of the current function
    NOTE: This will be the name of the function that called this function,
    so it serves to ensure we don't hardcode the function name in the step,
    but allow the step names to be refactored
    """
    return inspect.stack()[1].function


# All steps below have the Step signature


def simple_gen(ai: AI, dbs: DBs) -> List[Message]:
    """Run the AI on the main prompt and save the results"""
    messages = ai.start(setup_sys_prompt(dbs), get_prompt(dbs), step_name=curr_fn())
    to_files(messages[-1].content.strip(), dbs.workspace)
    return messages


def ask_feedback(ai: AI, dbs: DBs) -> List[dict]:
    """Ask the user for feedback on the generated program, and save it to the feedback
    file"""
    review = dbs.memory.get("review", None)

    print("The most recent review was: " + review)
    print()
    user_input = input("Press enter to use the review as feedback, or type a new one: ")

    dbs.input["feedback"] = user_input if user_input else review
    return []


def choice_input(
    ai: AI, dbs: DBs, choices: List[str], examples: List[tuple], text_input: str
) -> str:
    """Determine what the user selected from a menu of choices"""
    choices_str = "\n".join(choices)
    examples_str = ""
    # if examples and len(examples) > 0:
    #     examples_str = "Examples:\n" + \
    #         "\n".join(
    #             [f"Input: {sentence}\nOutput: [{{'C': '{label}'}}]" for sentence,
    #               label in examples])
    messages = ai.start(
        f"""You are a highly intelligent and accurate Multiclass Classification system.
        You take Passage as input and classify that as one of the following appropriate
        Categories:
        {choices_str}
        If the input does not belong to any of the categories, output None.
        Your output format is only [{{'C': Appropriate Category from the list of provided
        Categories}}] form, no other form.

        {examples_str}
        """,
        f"""Input: { text_input }
        Output:
        """,
        step_name=curr_fn(),
    )
    raw_result = messages[-1]["content"]
    try:
        result = json.loads(raw_result)
    except Exception as ex:
        print("Error parsing result: " + raw_result)
        print(ex)
        result = {}
    return result.get("C", None)


def choice_loop(ai: AI, dbs: DBs) -> List[dict]:
    """Ask the user to choose the steps to run, and keep looping until they say they're
    done"""

    # Get all the step keys that start with LOOP_BODY_
    loop_body_keys = [key for key in STEPS.keys() if key.startswith(LOOP_BODY_PREFIX)]

    # Keep looping until the user says they're done
    while True:
        body_steps = None
        while not body_steps:
            # Give the user the list to choose from, and ask them to choose
            print("Choose a step to run:")
            choices = []
            for i, key in enumerate(loop_body_keys):
                choice = f"{i+1}. {key.removeprefix(LOOP_BODY_PREFIX).title()}"
                print(choice)
                choices.append(choice)
            print()

            user_input = input(
                "Enter a number to choose a step, or enter 'done' to finish: "
            )
            print()

            # If they say they're done, break out of the loop
            if user_input == "done" or user_input == "quit":
                return []

            # If they enter a number, run the corresponding step
            if user_input.isdigit():
                step_index = int(user_input) - 1
                step_key = loop_body_keys[step_index]
                body_steps = STEPS.get(step_key, [])

            # If they enter something else, ask the AI to classify it
            else:
                user_input = choice_input(
                    ai,
                    dbs,
                    choices,
                    [("feeback", 1), ("quux", None), ("run", "Evaluate")],
                    user_input,
                )

                # If the AI says it doesn't understand, ask the user to clarify
                if user_input is None:
                    print("I don't understand. Please clarify.")
                    print()
                else:
                    step_index = int(user_input) - 1
                    step_key = loop_body_keys[step_index]
                    body_steps = STEPS.get(step_key, [])
                    break

        # Run the steps they chose
        for step in body_steps:
            messages = step(ai, dbs)
            dbs.logs[step.__name__] = json.dumps(messages)


def ask_feedback(ai: AI, dbs: DBs) -> List[dict]:
    """Ask the user for feedback on the generated program, and save it to the feedback
    file"""
    review = dbs.memory.get("review", None)

    print("The most recent review was: " + review)
    print()
    user_input = input("Press enter to use the review as feedback, or type a new one: ")

    dbs.input["feedback"] = user_input if user_input else review
    return []


def choice_input(
    ai: AI, dbs: DBs, choices: List[str], examples: List[tuple], text_input: str
) -> str:
    """Determine what the user selected from a menu of choices"""
    choices_str = "\n".join(choices)
    examples_str = ""
    # if examples and len(examples) > 0:
    #     examples_str = "Examples:\n" + \
    #         "\n".join(
    #             [f"Input: {sentence}\nOutput: [{{'C': '{label}'}}]" for sentence,
    #               label in examples])
    messages = ai.start(
        f"""You are a highly intelligent and accurate Multiclass Classification system.
        You take Passage as input and classify that as one of the following appropriate
        Categories:
        {choices_str}
        If the input does not belong to any of the categories, output None.
        Your output format is only [{{'C': Appropriate Category from the list of provided
        Categories}}] form, no other form.

        {examples_str}
        """,
        f"""Input: { text_input }
        Output:
        """,
        step_name=curr_fn(),
    )
    raw_result = messages[-1]["content"]
    try:
        result = json.loads(raw_result)
    except Exception as ex:
        print("Error parsing result: " + raw_result)
        print(ex)
        result = {}
    return result.get("C", None)


def choice_loop(ai: AI, dbs: DBs) -> List[dict]:
    """Ask the user to choose the steps to run, and keep looping until they say they're
    done"""

    # Get all the step keys that start with LOOP_BODY_
    loop_body_keys = [key for key in STEPS.keys() if key.startswith(LOOP_BODY_PREFIX)]

    # Keep looping until the user says they're done
    while True:
        body_steps = None
        while not body_steps:
            # Give the user the list to choose from, and ask them to choose
            print("Choose a step to run:")
            choices = []
            for i, key in enumerate(loop_body_keys):
                choice = f"{i+1}. {key.removeprefix(LOOP_BODY_PREFIX).title()}"
                print(choice)
                choices.append(choice)
            print()

            user_input = input(
                "Enter a number to choose a step, or enter 'done' to finish: "
            )
            print()

            # If they say they're done, break out of the loop
            if user_input == "done" or user_input == "quit":
                return []

            # If they enter a number, run the corresponding step
            if user_input.isdigit():
                step_index = int(user_input) - 1
                step_key = loop_body_keys[step_index]
                body_steps = STEPS.get(step_key, [])

            # If they enter something else, ask the AI to classify it
            else:
                user_input = choice_input(
                    ai,
                    dbs,
                    choices,
                    [("feeback", 1), ("quux", None), ("run", "Evaluate")],
                    user_input,
                )

                # If the AI says it doesn't understand, ask the user to clarify
                if user_input is None:
                    print("I don't understand. Please clarify.")
                    print()
                else:
                    step_index = int(user_input) - 1
                    step_key = loop_body_keys[step_index]
                    body_steps = STEPS.get(step_key, [])
                    break

        # Run the steps they chose
        for step in body_steps:
            messages = step(ai, dbs)
            dbs.logs[step.__name__] = json.dumps(messages)


def clarify(ai: AI, dbs: DBs) -> List[Message]:
    """
    Ask the user if they want to clarify anything and save the results to the workspace
    """
    messages: List[Message] = [ai.fsystem(dbs.preprompts["clarify"])]
    user_input = get_prompt(dbs)
    while True:
        messages = ai.next(messages, user_input, step_name=curr_fn())
        msg = messages[-1].content.strip()

        if msg == "Nothing more to clarify.":
            break

        if msg.lower().startswith("no"):
            print("Nothing more to clarify.")
            break

        print()
        user_input = input('(answer in text, or "c" to move on)\n')
        print()

        if not user_input or user_input == "c":
            print("(letting gpt-engineer make its own assumptions)")
            print()
            messages = ai.next(
                messages,
                "Make your own assumptions and state them explicitly before starting",
                step_name=curr_fn(),
            )
            print()
            return messages

        user_input += (
            "\n\n"
            "Is anything else unclear? If yes, only answer in the form:\n"
            "{remaining unclear areas} remaining questions.\n"
            "{Next question}\n"
            'If everything is sufficiently clear, only answer "Nothing more to clarify.".'
        )

    print()
    return messages


def gen_spec(ai: AI, dbs: DBs) -> List[Message]:
    """
    Generate a spec from the main prompt + clarifications and save the results to
    the workspace
    """
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fsystem(f"Instructions: {dbs.input['prompt']}"),
    ]

    messages = ai.next(messages, dbs.preprompts["spec"], step_name=curr_fn())

    dbs.memory["specification"] = messages[-1].content.strip()

    return messages


def respec(ai: AI, dbs: DBs) -> List[Message]:
    """Asks the LLM to review the specs so far and reiterate them if necessary"""
    messages = AI.deserialize_messages(dbs.logs[gen_spec.__name__])
    messages += [ai.fsystem(dbs.preprompts["respec"])]

    messages = ai.next(messages, step_name=curr_fn())
    messages = ai.next(
        messages,
        (
            "Based on the conversation so far, please reiterate the specification for "
            "the program. "
            "If there are things that can be improved, please incorporate the "
            "improvements. "
            "If you are satisfied with the specification, just write out the "
            "specification word by word again."
        ),
        step_name=curr_fn(),
    )

    dbs.memory["specification"] = messages[-1].content.strip()
    return messages


def gen_unit_tests(ai: AI, dbs: DBs) -> List[dict]:
    """
    Generate unit tests based on the specification, that should work.
    """
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fuser(f"Instructions: {dbs.input['prompt']}"),
        ai.fuser(f"Specification:\n\n{dbs.memory['specification']}"),
    ]

    messages = ai.next(messages, dbs.preprompts["unit_tests"], step_name=curr_fn())

    dbs.memory["unit_tests"] = messages[-1].content.strip()
    to_files(dbs.memory["unit_tests"], dbs.workspace)

    return messages


def gen_clarified_code(ai: AI, dbs: DBs) -> List[dict]:
    """Takes clarification and generates code"""
    messages = AI.deserialize_messages(dbs.logs[clarify.__name__])

    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
    ] + messages[
        1:
    ]  # skip the first clarify message, which was the original clarify priming prompt
    messages = ai.next(messages, dbs.preprompts["generate"], step_name=curr_fn())

    to_files(messages[-1].content.strip(), dbs.workspace)
    return messages


def gen_code_after_unit_tests(ai: AI, dbs: DBs) -> List[dict]:
    """Generates project code after unit tests have been produced"""
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fuser(f"Instructions: {dbs.input['prompt']}"),
        ai.fuser(f"Specification:\n\n{dbs.memory['specification']}"),
        ai.fuser(f"Unit tests:\n\n{dbs.memory['unit_tests']}"),
    ]
    messages = ai.next(messages, dbs.preprompts["generate"], step_name=curr_fn())
    to_files(messages[-1].content.strip(), dbs.workspace)
    return messages


def execute_entrypoint(ai: AI, dbs: DBs) -> List[dict]:
    command = dbs.workspace["run.sh"]

    print()
    print(
        colored(
            "Do you want to execute this code? (y/n)",
            "red",
        )
    )
    print()
    print(command)
    print()
    print("To execute, you can also press enter.")
    print()
    if input() not in ["", "y", "yes"]:
        print("Ok, not executing the code.")
        return []
    print("Executing the code...")
    print()
    print(
        colored(
            "Note: If it does not work as expected, consider running the code"
            + " in another way than above.",
            "green",
        )
    )
    print()
    print("You can press ctrl+c *once* to stop the execution.")
    print()

    p = subprocess.Popen("bash run.sh", shell=True, cwd=dbs.workspace.path)
    try:
        p.wait()
    except KeyboardInterrupt:
        print()
        print("Stopping execution.")
        print("Execution stopped.")
        p.kill()
        print()

    return []


def gen_entrypoint(ai: AI, dbs: DBs) -> List[dict]:
    messages = ai.start(
        system=(
            "You will get information about a codebase that is currently on disk in "
            "the current folder.\n"
            "From this you will answer with code blocks that includes all the necessary "
            "unix terminal commands to "
            "a) install dependencies "
            "b) run all necessary parts of the codebase (in parallel if necessary).\n"
            "Do not install globally. Do not use sudo.\n"
            "Do not explain the code, just give the commands.\n"
            "Do not use placeholders, use example values (like . for a folder argument) "
            "if necessary.\n"
        ),
        user="Information about the codebase:\n\n" + dbs.workspace["all_output.txt"],
        step_name=curr_fn(),
    )
    print()

    regex = r"```\S*\n(.+?)```"
    matches = re.finditer(regex, messages[-1].content.strip(), re.DOTALL)
    dbs.workspace["run.sh"] = "\n".join(match.group(1) for match in matches)
    return messages


def use_feedback(ai: AI, dbs: DBs):
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fuser(f"Instructions: {dbs.input['prompt']}"),
        ai.fassistant(
            dbs.workspace["all_output.txt"]
        ),  # reload previously generated code
    ]
    if dbs.input["feedback"]:
        messages = ai.next(messages, dbs.input["feedback"], step_name=curr_fn())
        to_files(messages[-1].content.strip(), dbs.workspace)
        return messages
    else:
        print(
            "No feedback was found in the input folder. Please create a file "
            + "called 'feedback' in the same folder as the prompt file."
        )
        exit(1)


def set_improve_filelist(ai: AI, dbs: DBs):
    """Sets the file list for files to work with in existing code mode."""
    ask_for_files(dbs.input)  # stores files as full paths.
    return []


def assert_files_ready(ai: AI, dbs: DBs):
    """Checks that the required files are present for headless
    improve code execution."""
    assert (
        "file_list.txt" in dbs.input
    ), "For auto_mode file_list.txt need to be in your project folder."
    assert "prompt" in dbs.input, "For auto_mode a prompt file must exist."
    return []


def get_improve_prompt(ai: AI, dbs: DBs):
    """
    Asks the user what they would like to fix.
    """

    dbs.input["prompt"] = input(
        "\nWhat do you need to improve with the selected files?\n"
    )

    confirm_str = f"""
    -----------------------------
    The following files will be used in the improvement process:
    {dbs.input["file_list.txt"]}

    The inserted prompt is the following:
    '{dbs.input['prompt']}'
    -----------------------------

    You can change these files in .gpteng folder ({dbs.input.path}) in your project
    before proceeding.

    Press enter to proceed with modifications.

    """
    input(confirm_str)
    return []


def improve_existing_code(ai: AI, dbs: DBs):
    """
    After the file list and prompt have been aquired, this function is called
    to sent the formatted prompt to the LLM.
    """

    files_info = get_code_strings(dbs.input)  # this only has file names not paths

    messages = [
        ai.fsystem(setup_sys_prompt_existing_code(dbs)),
        ai.fuser(f"Instructions: {dbs.input['prompt']}"),
    ]
    # Add files as input
    for file_name, file_str in files_info.items():
        code_input = format_file_to_input(file_name, file_str)
        messages.append(ai.fuser(f"{code_input}"))

    output_format_str = """
    Make sure the output of any files is in the following format where
    FILENAME is the file name including the file extension, and the file path.  Do not
    forget to include the file path.
    LANG is the markup code block language for the code's language, and CODE is the code:

    FILENAME
    ```LANG
    CODE
    ```
    """

    messages = ai.next(messages, output_format_str, step_name=curr_fn())
    # Maybe we should add another step called "replace" or "overwrite"
    overwrite_files(messages[-1].content.strip(), dbs)
    return messages


def fix_code(ai: AI, dbs: DBs):
    messages = AI.deserialize_messages(dbs.logs[gen_code_after_unit_tests.__name__])
    code_output = messages[-1].content.strip()
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fuser(f"Instructions: {dbs.input['prompt']}"),
        ai.fuser(code_output),
        ai.fsystem(dbs.preprompts["fix_code"]),
    ]
    messages = ai.next(
        messages, "Please fix any errors in the code above.", step_name=curr_fn()
    )
    to_files(messages[-1].content.strip(), dbs.workspace)
    return messages


def human_review(ai: AI, dbs: DBs):
    """Collects and stores human review of the code"""
    review = human_review_input()
    if review is not None:
        dbs.memory["review"] = review.to_json()  # type: ignore
    return []


LOOP_BODY_PREFIX = "loop_body_"


class Config(str, Enum):
    DEFAULT = "default"
    BENCHMARK = "benchmark"
    SIMPLE = "simple"
    TDD = "tdd"
    TDD_PLUS = "tdd+"
    CLARIFY = "clarify"
    RESPEC = "respec"
    EXECUTE_ONLY = "execute_only"
    EVALUATE = "evaluate"
    USE_FEEDBACK = "use_feedback"
    IMPROVE_CODE = "improve_code"
    EVAL_IMPROVE_CODE = "eval_improve_code"
    CHOICE_LOOP = "choice_loop"
    LOOP_BODY_FEEDBACK = LOOP_BODY_PREFIX + "feedback"
    LOOP_BODY_EVALUATE = LOOP_BODY_PREFIX + "evaluate"


# Define the steps to run for different configs
STEPS = {
    Config.DEFAULT: [
        clarify,
        gen_clarified_code,
        gen_entrypoint,
        execute_entrypoint,
        human_review,
    ],
    Config.BENCHMARK: [
        simple_gen,
        gen_entrypoint,
    ],
    Config.SIMPLE: [
        simple_gen,
        gen_entrypoint,
        execute_entrypoint,
    ],
    Config.TDD: [
        gen_spec,
        gen_unit_tests,
        gen_code_after_unit_tests,
        gen_entrypoint,
        execute_entrypoint,
        human_review,
    ],
    Config.TDD_PLUS: [
        gen_spec,
        gen_unit_tests,
        gen_code_after_unit_tests,
        fix_code,
        gen_entrypoint,
        execute_entrypoint,
        human_review,
    ],
    Config.CLARIFY: [
        clarify,
        gen_clarified_code,
        gen_entrypoint,
        execute_entrypoint,
        human_review,
    ],
    Config.RESPEC: [
        gen_spec,
        respec,
        gen_unit_tests,
        gen_code_after_unit_tests,
        fix_code,
        gen_entrypoint,
        execute_entrypoint,
        human_review,
    ],
    Config.USE_FEEDBACK: [use_feedback, gen_entrypoint, execute_entrypoint, human_review],
    Config.EXECUTE_ONLY: [execute_entrypoint],
    Config.EVALUATE: [execute_entrypoint, human_review],
    Config.IMPROVE_CODE: [
        set_improve_filelist,
        get_improve_prompt,
        improve_existing_code,
    ],
    Config.EVAL_IMPROVE_CODE: [assert_files_ready, improve_existing_code],
    Config.CHOICE_LOOP: [choice_loop],
    Config.LOOP_BODY_EVALUATE: [execute_entrypoint, human_review],
}
STEPS[Config.LOOP_BODY_FEEDBACK] = [ask_feedback] + STEPS[Config.USE_FEEDBACK]

# Future steps that can be added:
# run_tests_and_fix_files
# execute_entrypoint_and_fix_files_if_it_results_in_error
