#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .. import is_torch_available
from ..utils import logging as transformers_logging
from ..utils.import_utils import is_pygments_available
from .agent_types import AgentAudio, AgentImage, AgentText
from .default_tools import BASE_PYTHON_TOOLS, FinalAnswerTool, setup_default_tools
from .llm_engine import HfEngine, MessageRole
from .prompts import (
    DEFAULT_CODE_SYSTEM_PROMPT,
    SEARCHING_REACT_CODE_SYSTEM_PROMPT,
    DEFAULT_REACT_CODE_SYSTEM_PROMPT,
    DEFAULT_REACT_JSON_SYSTEM_PROMPT,
    PLAN_UPDATE_FINAL_PLAN_REDACTION,
    SYSTEM_PROMPT_FACTS,
    SYSTEM_PROMPT_FACTS_UPDATE,
    SYSTEM_PROMPT_PLAN,
    SYSTEM_PROMPT_PLAN_UPDATE,
    USER_PROMPT_FACTS_UPDATE,
    USER_PROMPT_PLAN,
    USER_PROMPT_PLAN_UPDATE,
)
from .python_interpreter import LIST_SAFE_MODULES, evaluate_python_code
from .tools import (
    DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
    Tool,
    get_tool_description_with_args,
    load_tool,
)


if is_pygments_available():
    from pygments import highlight
    from pygments.formatters import Terminal256Formatter
    from pygments.lexers import PythonLexer


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    bold_yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_red = "\x1b[31;1m"
    bold_white = "\x1b[37;1m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: format,
        logging.WARNING: bold_yellow + format + reset,
        31: reset + format + reset,
        32: green + format + reset,
        33: bold_white + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = transformers_logging.get_logger(__name__)
logger.propagate = False
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


def parse_json_blob(json_blob: str) -> Dict[str, str]:
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_blob = json_blob[first_accolade_index : last_accolade_index + 1].replace('\\"', "'")
        json_data = json.loads(json_blob, strict=False)
        return json_data
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place-4:place+5]}'."
        )
    except Exception as e:
        raise ValueError(f"Error in parsing the JSON blob: {e}")


def parse_code_blob(code_blob: str) -> str:
    try:
        pattern = r"```(?:py|python)?\n(.*?)\n```"
        match = re.search(pattern, code_blob, re.DOTALL)
        return match.group(1).strip()
    except Exception as e:
        raise ValueError(
            f"""
The code blob you used is invalid: due to the following error: {e}
This means that the regex pattern {pattern} was not respected: make sure to include code with the correct pattern, for instance:
Thoughts: Your thoughts
Code:
```py
# Your python code here
```<end_action>"""
        )


def parse_json_tool_call(json_blob: str) -> Tuple[str, Dict[str, str]]:
    json_blob = json_blob.replace("```json", "").replace("```", "")
    tool_call = parse_json_blob(json_blob)
    if "action" in tool_call and "action_input" in tool_call:
        return tool_call["action"], tool_call["action_input"]
    elif "action" in tool_call:
        return tool_call["action"], None
    else:
        raise ValueError(
            f"Missing keys: {[key for key in ['action', 'action_input'] if key not in tool_call]} in blob {tool_call}"
        )


def parse_text_tool_call(text: str) -> Tuple[str, Union[str, Dict[str, str]]]:
    """
    Expects a text in the format: 'Action:', 'Action input:', 'Observation:'. 'Action input:' contains a json string with input arguments.
    """
    try:
        if "Observation:" in text:
            text = text.split("Observation:")[0]
        if "Action:" in text:
            text = text.split("Action:")[1]
        tool_name, tool_input = text.split("Action input:")
        if "{" in tool_input:
            tool_input = parse_json_blob(tool_input)
        else:
            tool_input = tool_input.strip().replace('"', "")
        return tool_name.strip().replace('"', "").replace("\\", ""), tool_input
    except Exception as e:
        raise ValueError(
            f"Error in parsing the text tool call: {e}. Be sure to provide the correct format. DO NOT repeat your previous incorrect tool call."
        )


def to_text(input: Union[List[Dict[str, str]], Dict[str, str], str]) -> str:
    if isinstance(input, list):
        return "\n".join([m["content"] for m in input])
    elif isinstance(input, dict):
        return input["content"]
    else:
        return input


HUGGINGFACE_DEFAULT_TOOLS = {}
_tools_are_initialized = False


class Toolbox:
    """
    The toolbox contains all tools that the agent can perform operations with, as well as a few methods to
    manage them.

    Args:
        tools (`List[Tool]`):
            The list of tools to instantiate the toolbox with
        add_base_tools (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to add the tools available within `transformers` to the toolbox.
    """

    def __init__(self, tools: List[Tool], add_base_tools: bool = False):
        self._tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.add_base_tools()
        self._load_tools_if_needed()

    def add_base_tools(self, add_python_interpreter: bool = False):
        global _tools_are_initialized
        global HUGGINGFACE_DEFAULT_TOOLS
        if not _tools_are_initialized:
            HUGGINGFACE_DEFAULT_TOOLS = setup_default_tools(logger)
            _tools_are_initialized = True
        for tool in HUGGINGFACE_DEFAULT_TOOLS.values():
            if tool.name != "python_interpreter" or add_python_interpreter:
                self.add_tool(tool)
        self._load_tools_if_needed()

    @property
    def tools(self) -> Dict[str, Tool]:
        """Get all tools currently in the toolbox"""
        return self._tools

    def show_tool_descriptions(self, tool_description_template: str = None) -> str:
        """
        Returns the description of all tools in the toolbox

        Args:
            tool_description_template (`str`, *optional*):
                The template to use to describe the tools. If not provided, the default template will be used.
        """
        return "\n".join(
            [get_tool_description_with_args(tool, tool_description_template) for tool in self._tools.values()]
        )

    def add_tool(self, tool: Tool):
        """
        Adds a tool to the toolbox

        Args:
            tool (`Tool`):
                The tool to add to the toolbox.
        """
        if tool.name in self._tools:
            raise KeyError(f"Error: tool '{tool.name}' already exists in the toolbox.")
        self._tools[tool.name] = tool

    def remove_tool(self, tool_name: str):
        """
        Removes a tool from the toolbox

        Args:
            tool_name (`str`):
                The tool to remove from the toolbox.
        """
        if tool_name not in self._tools:
            raise KeyError(
                f"Error: tool {tool_name} not found in toolbox for removal, should be instead one of {list(self._tools.keys())}."
            )
        del self._tools[tool_name]

    def update_tool(self, tool: Tool):
        """
        Updates a tool in the toolbox according to its name.

        Args:
            tool (`Tool`):
                The tool to update to the toolbox.
        """
        if tool.name not in self._tools:
            raise KeyError(
                f"Error: tool {tool.name} not found in toolbox for update, should be instead one of {list(self._tools.keys())}."
            )
        self._tools[tool.name] = tool

    def clear_toolbox(self):
        """Clears the toolbox"""
        self._tools = {}

    def _load_tools_if_needed(self):
        for name, tool in self._tools.items():
            if not isinstance(tool, Tool):
                task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
                self._tools[name] = load_tool(task_or_repo_id)

    def __repr__(self):
        toolbox_description = "Toolbox contents:\n"
        for tool in self._tools.values():
            toolbox_description += f"\t{tool.name}: {tool.description}\n"
        return toolbox_description


class AgentError(Exception):
    """Base class for other agent-related exceptions"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class AgentParsingError(AgentError):
    """Exception raised for errors in parsing in the agent"""

    pass


class AgentExecutionError(AgentError):
    """Exception raised for errors in execution in the agent"""

    pass


class AgentMaxIterationsError(AgentError):
    """Exception raised for errors in execution in the agent"""

    pass


class AgentGenerationError(AgentError):
    """Exception raised for errors in generation in the agent"""

    pass


def format_prompt_with_tools(toolbox: Toolbox, prompt_template: str, tool_description_template: str) -> str:
    tool_descriptions = toolbox.show_tool_descriptions(tool_description_template)
    prompt = prompt_template.replace("<<tool_descriptions>>", tool_descriptions)
    if "<<tool_names>>" in prompt:
        tool_names = [f"'{tool_name}'" for tool_name in toolbox.tools.keys()]
        prompt = prompt.replace("<<tool_names>>", ", ".join(tool_names))
    return prompt


def format_prompt_with_imports(prompt_template: str, authorized_imports: List[str]) -> str:
    if "<<authorized_imports>>" not in prompt_template:
        raise AgentError("Tag '<<authorized_imports>>' should be provided in the prompt.")
    return prompt_template.replace("<<authorized_imports>>", str(authorized_imports))


class Agent:
    def __init__(
        self,
        tools: Union[List[Tool], Toolbox],
        llm_engine: Callable = HfEngine(),
        system_prompt=DEFAULT_REACT_JSON_SYSTEM_PROMPT,
        tool_description_template=None,
        additional_args={},
        max_iterations: int = 6,
        tool_parser=parse_json_tool_call,
        add_base_tools: bool = False,
        verbose: int = 0,
        memory_verbose: bool = False,
    ):
        self.agent_name = self.__class__.__name__
        self.llm_engine = llm_engine
        self.system_prompt_template = system_prompt
        self.tool_description_template = (
            tool_description_template if tool_description_template else DEFAULT_TOOL_DESCRIPTION_TEMPLATE
        )
        self.additional_args = additional_args
        self.max_iterations = max_iterations
        self.logger = logger
        self.tool_parser = tool_parser

        if isinstance(tools, Toolbox):
            self._toolbox = tools
            if add_base_tools:
                if not is_torch_available():
                    raise ImportError("Using the base tools requires torch to be installed.")

                self._toolbox.add_base_tools(add_python_interpreter=(self.__class__ == ReactJsonAgent))
        else:
            self._toolbox = Toolbox(tools, add_base_tools=add_base_tools)
        self._toolbox.add_tool(FinalAnswerTool())

        self.system_prompt = format_prompt_with_tools(
            self._toolbox, self.system_prompt_template, self.tool_description_template
        )
        self.prompt = None
        self.logs = []
        self.task = None
        self.memory_verbose = memory_verbose

        if verbose == 0:
            logger.setLevel(logging.WARNING)
        elif verbose == 1:
            logger.setLevel(logging.INFO)
        elif verbose == 2:
            logger.setLevel(logging.DEBUG)

    @property
    def toolbox(self) -> Toolbox:
        """Get the toolbox currently available to the agent"""
        return self._toolbox

    def initialize_for_run(self):
        self.token_count = 0
        self.system_prompt = format_prompt_with_tools(
            self._toolbox,
            self.system_prompt_template,
            self.tool_description_template,
        )
        if hasattr(self, "authorized_imports"):
            self.system_prompt = format_prompt_with_imports(
                self.system_prompt, list(set(LIST_SAFE_MODULES) | set(self.authorized_imports))
            )
        self.logs = [{"system_prompt": self.system_prompt, "task": self.task}]
        self.logger.warn("======== New task ========")
        self.logger.log(33, self.task)
        self.logger.debug("System prompt is as follows:")
        self.logger.debug(self.system_prompt)

    def write_inner_memory_from_logs(self, summary_mode: Optional[bool] = False, is_mistral: bool = False) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the logs into a series of messages
        that can be used as input to the LLM.
        """
        if not is_mistral:
            prompt_message = {"role": MessageRole.SYSTEM, "content": self.logs[0]["system_prompt"]}
        else:
            prompt_message = {"role": MessageRole.USER, "content": self.logs[0]["system_prompt"].replace("<<tool_descriptions>>", "")}

        task_message = {
            "role": MessageRole.USER,
            "content": "Task: " + self.logs[0]["task"],
        }
        if summary_mode:
            memory = [task_message]
        else:
            memory = [prompt_message, task_message]
        for i, step_log in enumerate(self.logs[1:]):
            if "llm_output" in step_log and not summary_mode:
                thought_message = {"role": MessageRole.ASSISTANT, "content": step_log["llm_output"].strip()}
                memory.append(thought_message)
            if "facts" in step_log:
                thought_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": "[FACTS LIST]:\n" + step_log["facts"].strip(),
                }
                memory.append(thought_message)

            if "plan" in step_log and not summary_mode:
                thought_message = {"role": MessageRole.ASSISTANT, "content": "[PLAN]:\n" + step_log["plan"].strip()}
                memory.append(thought_message)

            if "tool_call" in step_log and summary_mode:
                tool_call_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": f"[STEP {i} TOOL CALL]: " + str(step_log["tool_call"]).strip(),
                }
                memory.append(tool_call_message)

            if "task" in step_log:
                tool_call_message = {
                    "role": MessageRole.USER,
                    "content": "New task:\n" + step_log["task"],
                }
                memory.append(tool_call_message)

            if "error" in step_log or "observation" in step_log:
                if "error" in step_log:
                    message_content = (
                        f"[OUTPUT OF STEP {i}] Error: "
                        + str(step_log["error"])
                        + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                    )
                elif "observation" in step_log:
                    message_content = f"[OUTPUT OF STEP {i}] Observation:\n{step_log['observation']}"
                tool_response_message = {"role": MessageRole.TOOL_RESPONSE, "content": message_content}
                memory.append(tool_response_message)

        return memory

    def get_succinct_logs(self):
        return [{key: value for key, value in log.items() if key != "agent_memory"} for log in self.logs]

    def extract_action(self, llm_output: str, split_token: str) -> str:
        """
        Parse action from the LLM output

        Args:
            llm_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = llm_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception as e:
            self.logger.error(e, exc_info=1)
            raise AgentParsingError(
                f"Error: No '{split_token}' token provided in your output.\nYour output:\n{llm_output}\n. Be sure to include an action, prefaced with '{split_token}'!"
            )
        return rationale, action

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.toolbox).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        if tool_name not in self.toolbox.tools:
            error_msg = f"Error: unknown tool {tool_name}, should be instead one of {list(self.toolbox.tools.keys())}."
            self.logger.error(error_msg, exc_info=1)
            raise AgentExecutionError(error_msg)

        try:
            if isinstance(arguments, str):
                observation = self.toolbox.tools[tool_name](arguments)
            else:
                for key, value in arguments.items():
                    # if the value is the name of a state variable like "image.png", replace it with the actual value
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                observation = self.toolbox.tools[tool_name](**arguments)
            return observation
        except Exception as e:
            raise AgentExecutionError(
                f"Error in tool call execution: {e}\nYou should only use this tool with a correct input.\n"
                f"As a reminder, this tool's description is the following:\n{get_tool_description_with_args(self.toolbox.tools[tool_name])}"
            )

    def log_code_action(self, code_action: str) -> None:
        self.logger.warning("==== Agent is executing the code below:")
        if is_pygments_available():
            self.logger.log(
                31, highlight(code_action, PythonLexer(ensurenl=False), Terminal256Formatter(style="nord"))
            )
        else:
            self.logger.log(31, code_action)
        self.logger.warning("====")

    def run(self, **kwargs):
        """To be implemented in the child class"""
        raise NotImplementedError


class CodeAgent(Agent):
    """
    A class for an agent that solves the given task using a single block of code. It plans all its actions, then executes all in one shot.
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Callable = HfEngine(),
        system_prompt: str = DEFAULT_CODE_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        additional_authorized_imports: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            **kwargs,
        )

        if not is_pygments_available():
            transformers_logging.warning_once(
                logger,
                "pygments isn't installed. Installing pygments will enable color syntax highlighting in the "
                "CodeAgent.",
            )

        self.python_evaluator = evaluate_python_code
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(LIST_SAFE_MODULES) | set(self.additional_authorized_imports))
        self.system_prompt = self.system_prompt.replace("<<authorized_imports>>", str(self.authorized_imports))

    def parse_code_blob(self, result: str) -> str:
        """
        Override this method if you want to change the way the code is
        cleaned in the `run` method.
        """
        return parse_code_blob(result)

    def run(self, task: str, return_generated_code: bool = False, is_mistral: bool = False, **kwargs):
        """
        Runs the agent for the given task.

        Args:
            task (`str`): The task to perform
            return_generated_code (`bool`, *optional*, defaults to `False`): Whether to return the generated code instead of running it
            kwargs (additional keyword arguments, *optional*):
                Any keyword argument to send to the agent when evaluating the code.

        Example:

        ```py
        from transformers.agents import CodeAgent, PythonInterpreterTool

        python_interpreter = PythonInterpreterTool()
        agent = CodeAgent(tools=[python_interpreter])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        self.task = task
        if len(kwargs) > 0:
            self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        self.state = kwargs.copy()
        self.initialize_for_run()

        # Run LLM
        if not is_mistral:
            prompt_message = {"role": MessageRole.SYSTEM, "content": self.system_prompt}
        else:
            prompt_message = {"role": MessageRole.USER, "content": self.system_prompt}

        task_message = {
            "role": MessageRole.USER,
            "content": "Task: " + self.task,
        }

        self.prompt = [prompt_message, task_message]
        self.logger.info("====Executing with this prompt====")
        self.logger.info(self.prompt)
        llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>"])

        if return_generated_code:
            return llm_output

        # Parse
        try:
            _, code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(
                f"Error in extracting action, trying to parse the whole output as code. Error trace: {e}"
            )
            code_action = llm_output

        try:
            code_action = self.parse_code_blob(code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Be sure to provide correct code"
            self.logger.error(error_msg, exc_info=1)
            return error_msg

        # Execute
        self.log_code_action(code_action)
        try:
            available_tools = {**BASE_PYTHON_TOOLS.copy(), **self.toolbox.tools}
            output = self.python_evaluator(
                code_action,
                static_tools=available_tools,
                custom_tools={},
                state=self.state,
                authorized_imports=self.authorized_imports,
            )
            self.logger.info(self.state["print_outputs"])
            return output
        except Exception as e:
            error_msg = f"Error in execution: {e}. Be sure to provide correct code."
            self.logger.error(error_msg, exc_info=1)
            return error_msg


class ReactAgent(Agent):
    """
    This agent that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of thinking and acting.
    The action will be parsed from the LLM output: it consists in calls to tools from the toolbox, with arguments chosen by the LLM engine.
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Callable = HfEngine(),
        system_prompt: str = DEFAULT_REACT_CODE_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            **kwargs,
        )
        self.planning_interval = planning_interval

    def provide_final_answer(self, task, is_mistral=False) -> str:
        """
        This method provides a final answer to the task, based on the logs of the agent's interactions.
        """
        if not is_mistral:
            self.prompt = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": "An agent tried to answer an user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
                }
            ]
            self.prompt += self.write_inner_memory_from_logs(is_mistral=is_mistral)[1:]
            self.prompt += [
                {
                    "role": MessageRole.USER,
                    "content": f"Based on the above, please provide an answer to the following user request:\n{task}",
                }
            ]
        else:
            self.prompt = [
                {
                    "role": MessageRole.USER,
                    "content": "An agent tried to answer an user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
                }
            ]
            self.prompt += self.write_inner_memory_from_logs(is_mistral=is_mistral)[1:]
            self.prompt += [
                {
                    "role": MessageRole.USER,
                    "content": f"Based on the above, please provide an answer to the following user request:\n{task}",
                }
            ]
        try:
            return self.llm_engine(self.prompt)
        except Exception as e:
            return f"Error in generating final llm output: {e}."

    def run(self, task: str, stream: bool = False, reset: bool = True, is_mistral: bool = False, **kwargs):
        """
        Runs the agent for the given task.

        Args:
            task (`str`): The task to perform

        Example:
        ```py
        from transformers.agents import ReactCodeAgent
        agent = ReactCodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        self.task = task
        if len(kwargs) > 0:
            self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        self.state = kwargs.copy()
        if reset:
            self.initialize_for_run()
        else:
            self.logs.append({"task": task})
        if stream:
            return self.stream_run(task, is_mistral=is_mistral)
        else:
            return self.direct_run(task, is_mistral=is_mistral)

    def stream_run(self, task: str, is_mistral: bool = False):
        """
        Runs the agent in streaming mode, yielding steps as they are executed: should be launched only in the `run` method.
        """
        final_answer = None
        iteration = 0
        step_observation_logs = [] # >>>
        while final_answer is None and iteration < self.max_iterations:
            try:
                step_logs, ob = self.step()
                step_observation_logs.append(ob) # >>>
                if "final_answer" in step_logs:
                    final_answer = step_logs["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                self.logs[-1]["error"] = e
            finally:
                iteration += 1
                yield self.logs[-1]

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task, is_mistral=is_mistral)
            final_step_log["final_answer"] = final_answer
            yield final_step_log

        yield final_answer, step_observation_logs # >>>

    def direct_run(self, task: str, is_mistral: bool = False):
        """
        Runs the agent in direct mode, returning outputs only at the end: should be launched only in the `run` method.
        """
        final_answer = None
        iteration = 0
        step_observation_logs = [] # >>>
        sm_logs = [] # >>>
        while final_answer is None and iteration < self.max_iterations:
            try:
                if self.planning_interval is not None and iteration % self.planning_interval == 0:
                    self.planning_step(task, is_first_step=(iteration == 0), iteration=iteration, is_mistral=is_mistral)
                step_logs, ob, sm = self.step()
                step_observation_logs.append(ob) # >>>
                sm_logs.append(sm) # >>>
                if "final_answer" in step_logs:
                    final_answer = step_logs["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                self.logs[-1]["error"] = e
            finally:
                iteration += 1

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            
            final_answer = self.provide_final_answer(task, is_mistral=is_mistral)
            if final_answer is not None:
                self.logger.warning('>>> Final Answer Due to Max iter >>')
                self.logger.log(32, final_answer)
                final_step_log["final_answer"] = final_answer

        return final_answer, step_observation_logs, sm_logs # >>>

    def planning_step(self, task, is_first_step: bool = False, iteration: int = None, is_mistral: bool = False):
        """
        Used periodically by the agent to plan the next steps to reach the objective.

        Args:
            task (`str`): The task to perform
            is_first_step (`bool`): If this step is not the first one, the plan should be an update over a previous plan.
            iteration (`int`): The number of the current step, used as an indication for the LLM.
        """
        if is_first_step:
            if not is_mistral:
                message_prompt_facts = {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT_FACTS}
                message_system_prompt_plan = {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT_PLAN}
            else:
                message_prompt_facts = {"role": MessageRole.USER, "content": SYSTEM_PROMPT_FACTS}
                message_system_prompt_plan = {"role": MessageRole.USER, "content": SYSTEM_PROMPT_PLAN}
            message_prompt_task = {
                "role": MessageRole.USER,
                "content": f"""Here is the task:
```
{task}
```
Now begin!""",
            }

            answer_facts = self.llm_engine([message_prompt_facts, message_prompt_task])

            message_user_prompt_plan = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_PLAN.format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(self.tool_description_template),
                    answer_facts=answer_facts,
                ),
            }
            answer_plan = self.llm_engine(
                [message_system_prompt_plan, message_user_prompt_plan], stop_sequences=["<end_plan>"]
            )

            final_plan_redaction = f"""Here is the plan of action that I will follow to solve the task:
```
{answer_plan}
```"""
            final_facts_redaction = f"""Here are the facts that I know so far:
```
{answer_facts}
```""".strip()
            self.logs.append({"plan": final_plan_redaction, "facts": final_facts_redaction})
            self.logger.debug("===== Initial plan: =====")
            self.logger.debug(final_plan_redaction)
        else:  # update plan
            agent_memory = self.write_inner_memory_from_logs(
                summary_mode=False,
                is_mistral=is_mistral
            )  # This will not log the plan but will log facts

            # Redact updated facts
            if not is_mistral:
                facts_update_system_prompt = {
                    "role": MessageRole.SYSTEM,
                    "content": SYSTEM_PROMPT_FACTS_UPDATE,
                }
            else:
                facts_update_system_prompt = {
                    "role": MessageRole.USER,
                    "content": SYSTEM_PROMPT_FACTS_UPDATE,
                }
            facts_update_message = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_FACTS_UPDATE,
            }
            facts_update = self.llm_engine([facts_update_system_prompt] + agent_memory + [facts_update_message])

            # Redact updated plan
            if not is_mistral:
                plan_update_message = {
                    "role": MessageRole.SYSTEM,
                    "content": SYSTEM_PROMPT_PLAN_UPDATE.format(task=task),
                }
            else:
                plan_update_message = {
                    "role": MessageRole.USER,
                    "content": SYSTEM_PROMPT_PLAN_UPDATE.format(task=task),
                }
            plan_update_message_user = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_PLAN_UPDATE.format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(self.tool_description_template),
                    facts_update=facts_update,
                    remaining_steps=(self.max_iterations - iteration),
                ),
            }
            plan_update = self.llm_engine(
                [plan_update_message] + agent_memory + [plan_update_message_user], stop_sequences=["<end_plan>"]
            )

            # Log final facts and plan
            final_plan_redaction = PLAN_UPDATE_FINAL_PLAN_REDACTION.format(task=task, plan_update=plan_update)
            final_facts_redaction = f"""Here is the updated list of the facts that I know:
```
{facts_update}
```"""
            self.logs.append({"plan": final_plan_redaction, "facts": final_facts_redaction})
            self.logger.debug("===== Updated plan: =====")
            self.logger.debug(final_plan_redaction)


class ReactJsonAgent(ReactAgent):
    """
    This agent that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of thinking and acting.
    The tool calls will be formulated by the LLM in JSON format, then parsed and executed.
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Callable = HfEngine(),
        system_prompt: str = DEFAULT_REACT_JSON_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            planning_interval=planning_interval,
            **kwargs,
        )

    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.prompt = agent_memory
        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        self.logger.info("===== Calling LLM with this last message: =====")
        self.logger.info(self.prompt[-1])

        try:
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"])
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output

        # Parse
        self.logger.debug("===== Extracting action =====")
        rationale, action = self.extract_action(llm_output=llm_output, split_token="Action:")

        try:
            tool_name, arguments = self.tool_parser(action)
        except Exception as e:
            raise AgentParsingError(f"Could not parse the given action: {e}.")

        current_step_logs["rationale"] = rationale
        current_step_logs["tool_call"] = {"tool_name": tool_name, "tool_arguments": arguments}

        # Execute
        self.logger.warning(f"Calling tool: '{tool_name}' with arguments: {arguments}")
        if tool_name == "final_answer":
            if isinstance(arguments, dict):
                if "answer" in arguments:
                    answer = arguments["answer"]
                    if (
                        isinstance(answer, str) and answer in self.state.keys()
                    ):  # if the answer is a state variable, return the value
                        answer = self.state[answer]
                else:
                    answer = arguments
            else:
                answer = arguments
            current_step_logs["final_answer"] = answer
            return current_step_logs
        else:
            observation = self.execute_tool_call(tool_name, arguments)
            observation_type = type(observation)
            if observation_type == AgentText:
                updated_information = str(observation).strip()
            else:
                # TODO: observation naming could allow for different names of same type
                if observation_type == AgentImage:
                    observation_name = "image.png"
                elif observation_type == AgentAudio:
                    observation_name = "audio.mp3"
                else:
                    observation_name = "object.object"

                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."

            self.logger.info(updated_information)
            current_step_logs["observation"] = updated_information
            return current_step_logs


class ReactCodeAgent(ReactAgent):
    """
    This agent that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of thinking and acting.
    The tool calls will be formulated by the LLM in code format, then parsed and executed.
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Callable = HfEngine(),
        system_prompt: str = DEFAULT_REACT_CODE_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        is_mistral: bool = False,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            planning_interval=planning_interval,
            **kwargs,
        )

        if not is_pygments_available():
            transformers_logging.warning_once(
                logger,
                "pygments isn't installed. Installing pygments will enable color syntax highlighting in the "
                "ReactCodeAgent.",
            )

        self.python_evaluator = evaluate_python_code
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(LIST_SAFE_MODULES) | set(self.additional_authorized_imports))
        self.system_prompt = self.system_prompt.replace("<<authorized_imports>>", str(self.authorized_imports))
        self.custom_tools = {}
        self.is_mistral = is_mistral

    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs(is_mistral=self.is_mistral)

        self.prompt = agent_memory.copy()

        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        self.logger.info("===== Calling LLM with these last messages: =====")
        self.logger.info(self.prompt[-2:])

        try:
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"])
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")

        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output

        # Parse
        self.logger.debug("===== Extracting action =====")
        try:
            rationale, raw_code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(f"Error in extracting action, trying to parse the whole output. Error trace: {e}")
            rationale, raw_code_action = llm_output, llm_output

        try:
            code_action = parse_code_blob(raw_code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code"
            raise AgentParsingError(error_msg)

        current_step_logs["rationale"] = rationale
        current_step_logs["tool_call"] = {"tool_name": "code interpreter", "tool_arguments": code_action}

        # Execute
        self.log_code_action(code_action)
        try:
            result = self.python_evaluator(
                code_action,
                static_tools={
                    **BASE_PYTHON_TOOLS.copy(),
                    **self.toolbox.tools,
                },
                custom_tools=self.custom_tools,
                state=self.state,
                authorized_imports=self.authorized_imports,
            )
            information = self.state["print_outputs"]
            self.logger.warning("Print outputs:")
            self.logger.log(32, information)
            current_step_logs["observation"] = information
        except Exception as e:
            error_msg = f"Code execution failed due to the following error:\n{str(e)}"
            if "'dict' object has no attribute 'read'" in str(e):
                error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
            raise AgentExecutionError(error_msg)
        for line in code_action.split("\n"):
            if line[: len("final_answer")] == "final_answer":
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, result)
                current_step_logs["final_answer"] = result
        
        obs = current_step_logs["observation"] if "observation" in current_step_logs else ''
        return current_step_logs, obs # >>
    

###########################################
TAVILY_API_URL = "https://api.tavily.com"
import json
from typing import Dict, List, Optional

import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env

from typing import List, Optional, Sequence, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents import AgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import ToolsRenderer, render_text_description

class TavilySearchAPIWrapper(BaseModel):

    """Wrapper for Tavily Search API."""

    tavily_api_key: SecretStr
    context_str_limit: int = 800

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[str] = "advanced",
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
        include_images: Optional[bool] = False,
    ) -> Dict:
        query = query.ljust(5, ' ')
        params = {
            "api_key": self.tavily_api_key.get_secret_value(),
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
        }
        response = requests.post(
            # type: ignore
            f"{TAVILY_API_URL}/search",
            json=params,
        )
        response.raise_for_status()
        return response.json()

    def results(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[str] = "advanced",
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
        include_images: Optional[bool] = False,
    ) -> List[Dict]:
        """Run query through Tavily Search and return metadata.

        Args:
            query: The query to search for.
            max_results: The maximum number of results to return.
            search_depth: The depth of the search. Can be "basic" or "advanced".
            include_domains: A list of domains to include in the search.
            exclude_domains: A list of domains to exclude from the search.
            include_answer: Whether to include the answer in the results.
            include_raw_content: Whether to include the raw content in the results.
            include_images: Whether to include images in the results.
        Returns:
            query: The query that was searched for.
            follow_up_questions: A list of follow up questions.
            response_time: The response time of the query.
            answer: The answer to the query.
            images: A list of images.
            results: A list of dictionaries containing the results:
                title: The title of the result.
                url: The url of the result.
                content: The content of the result.
                score: The score of the result.
                raw_content: The raw content of the result.
        """  # noqa: E501
        query = query.ljust(5, ' ')
        raw_search_results = self.raw_results(
            query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )
        # return raw_search_results["results"]
        return self.clean_results(raw_search_results["results"])

    def results_content_only(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[str] = "advanced",
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
        include_images: Optional[bool] = False,
    ) -> List[Dict]:
        """Run query through Tavily Search and return metadata.

        Args:
            query: The query to search for.
            max_results: The maximum number of results to return.
            search_depth: The depth of the search. Can be "basic" or "advanced".
            include_domains: A list of domains to include in the search.
            exclude_domains: A list of domains to exclude from the search.
            include_answer: Whether to include the answer in the results.
            include_raw_content: Whether to include the raw content in the results.
            include_images: Whether to include images in the results.
        Returns:
            query: The query that was searched for.
            follow_up_questions: A list of follow up questions.
            response_time: The response time of the query.
            answer: The answer to the query.
            images: A list of images.
            results: A list of dictionaries containing the results:
                title: The title of the result.
                url: The url of the result.
                content: The content of the result.
                score: The score of the result.
                raw_content: The raw content of the result.
        """  # noqa: E501
        query = query.ljust(5, ' ')
        raw_search_results = self.raw_results(
            query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )
        # return raw_search_results["results"]
        return self.clean_results_content_only(raw_search_results["results"])

    async def raw_results_async(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[str] = "advanced",
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
        include_images: Optional[bool] = False,
    ) -> Dict:
        """Get results from the Tavily Search API asynchronously."""

        # Function to perform the API call
        query = query.ljust(5, ' ')
        async def fetch() -> str:
            params = {
                "api_key": self.tavily_api_key.get_secret_value(),
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{TAVILY_API_URL}/search", json=params) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()
        return json.loads(results_json_str)

    async def results_async(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[str] = "advanced",
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
        include_images: Optional[bool] = False,
    ) -> List[Dict]:
        query = query.ljust(5, ' ')
        results_json = await self.raw_results_async(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )
        return self.clean_results(results_json["results"])

    def clean_results(self, results: List[Dict], query: str ='') -> List[Dict]:
        """Clean results from Tavily Search API and truncate content."""
        truncated_results = []

        for result in results:

            if result["url"].endswith('.pdf'):
                print('>>>>>>HERE IS PDF>>>>>>>')
                continue

            if result['raw_content'] is not None:
                raw_content = result['raw_content']
                if len(raw_content) > self.context_str_limit:
                    raw_content = raw_content[:self.context_str_limit]
                assert len(raw_content) <= self.context_str_limit
                truncated_results.append({"title": result["title"],
                                        "url": result["url"], 
                                        "raw_content": raw_content})
            else:
                content = result['content']
                if len(content) > self.context_str_limit:
                    content = content[:self.context_str_limit]
                assert len(content) <= self.context_str_limit
                truncated_results.append({"title": result["title"],
                                        "url": result["url"], 
                                        "content": content})
        return truncated_results
    
    def clean_results_content_only(self, results: List[Dict], query: str ='') -> List[Dict]:
        """Clean results from Tavily Search API and truncate content."""
        truncated_results = []

        for result in results:

            if result["url"].endswith('.pdf'):
                print('>>>>>>HERE IS PDF>>>>>>>')
                continue

            if result['raw_content'] is not None:
                raw_content = result['raw_content']
                if len(raw_content) > self.context_str_limit:
                    raw_content = raw_content[:self.context_str_limit]
                assert len(raw_content) <= self.context_str_limit
                truncated_results.append({"raw_content": raw_content})
            else:
                content = result['content']
                if len(content) > self.context_str_limit:
                    content = content[:self.context_str_limit]
                assert len(content) <= self.context_str_limit
                truncated_results.append({"content": content})
        return truncated_results

class TavilySearchHuggingfaceTool(Tool):
    """Tool that queries the Tavily Search API and gets back json."""

    name: str = "tavily_search_results"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    inputs = {
        "task": {
            "type": "text",
            "description": "the text input of search engine",
        }
    }
    output_type = "text"
    max_results: int = 5
    search_depth: str = 'basic'
    context_str_limit: int = 5000

    def truncate_context(self, data: List[Dict], max_length=5000):
        """
        Truncates the text content in each dictionary of the given list
        if it exceeds the specified maximum length.

        Args:
        - data (List[Dict]): List of dictionaries where each dictionary contains 'url' and 'context' keys.
        - max_length (int): Maximum allowed length for the context string.

        Returns:
        - List[Dict]: Modified list with truncated context strings.
        """
        # data_copy = data.copy()
        # for i in range(len(data_copy)):
        #     if 'content' in data_copy[i]:
        #         context = data_copy[i]['content']
        #         if len(context) > max_length:
        #             data_copy[i]['content'] = context[:max_length]
        # return data_copy
        for item in data:
            if 'content' in item.keys():
                context = item['content']
                if len(context) > max_length:
                    item['content'] = context[:max_length]
        return data

    def forward(
        self,
        query: str
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            self.api_wrapper = TavilySearchAPIWrapper()
            query = query.ljust(5, '?') 
            results = self.api_wrapper.results_content_only( # results
                query,
                self.max_results,
                search_depth=self.search_depth,
            )
            truncated_results = self.truncate_context(results, self.context_str_limit)
            return truncated_results

        except Exception as e:
            return repr(e)

SEARCHING_REACT_SYSTEM_PROMPT = """
Your task is to conduct a series of sequential searches. 
After each search, assess the results to decide whether further information is needed or if the search can conclude.

Use the following format:

Task: the input question you must answer
Thought: you should always think about what to do
Search: the query to search.
Observation: the result of the search
... (this Thought/Search/Observation can repeat 3 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, this should be in very detailed and include the summarization of all observation above. 

Here are a few examples:
---
Task: What is the weather like in New York today?
Thought: I should first search for the weather in New York today.
Search: weather in New York today
Observation: The weather in New York today is sunny with a high of 75°F and a low of 60°F.
Thought: Now I know the weather in New York today.
Final Answer: The weather in New York today is sunny with a high of 75°F and a low of 60°F.
---
Task: I plan to go to Hainan for 3 days. Since I particularly like Hainan's Wenchang chicken, I want to know which scenic spot has the best taste and the highest sales of Wenchang chicken. Please help me plan it.
Thought: I should first search for scenic spots in Hainan that offer Wenchang chicken.
Search: scenic spots in Hainan that offer Wenchang chicken
Observation: Wenchang City, Sanya, Tongguling, Wenchang Aerospace City
Thought: Now I know the scenic spots in Hainan that offer Wenchang chicken. I should now look at the ratings and reviews of Wenchang chicken at each scenic spot to identify the ones with the best taste and highest sales.
Search: ratings and reviews of Wenchang chicken at Wenchang City
Observation: We chose Wangji Wenchang Chicken Food City, which has a low score but very sincere evaluation. It turns out that food writers are very good at reading reviews. The chicken here is delicious.
Search: ratings and reviews of Wenchang chicken at Sanya
Observation: The First Market is recommended. It is crowded every day. Not only tourists come here to choose seafood, but even Sanya locals come to the First Market to buy seafood and daily consumption.
Thought: Now I know the ratings and reviews of Wenchang chicken at Wenchang City and Sanya. The First Market is recommended. I should now plan a three-day itinerary for my Hainan trip based on the scenic spots in Hainan.
Final Answer: Based on the search results, I recommend the following three-day itinerary for Hainan Sanya, focusing on tasting the famous local Wenchang chicken.
---
<<authorized_imports>>
Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""


class ReactCodeSearchAgent(ReactAgent):
    """
    This agent that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of thinking and acting.
    The tool calls will be formulated by the LLM in code format, then parsed and executed.
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Callable = HfEngine(),
        system_prompt: str = SEARCHING_REACT_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        is_mistral: bool = False,
        search_tool: Optional[Tool] = TavilySearchHuggingfaceTool(),
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            planning_interval=planning_interval,
            **kwargs,
        )

        if not is_pygments_available():
            transformers_logging.warning_once(
                logger,
                "pygments isn't installed. Installing pygments will enable color syntax highlighting in the "
                "ReactCodeAgent.",
            )

        self.python_evaluator = evaluate_python_code
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(LIST_SAFE_MODULES) | set(self.additional_authorized_imports))
        self.system_prompt = self.system_prompt.replace("<<authorized_imports>>", str(self.authorized_imports))
        self.custom_tools = {}
        self.is_mistral = is_mistral
        self.search_tool = search_tool 


    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs(is_mistral=self.is_mistral)

        self.prompt = agent_memory.copy()

        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        self.logger.info("===== Calling LLM with these last messages: =====")
        self.logger.info(self.prompt[-2:])

        try:
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"])
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")

        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output
        
        #########################################################################
        search_flag = False
        self.logger.debug("===== Extracting action =====")
        try:
            if "Search:" not in llm_output:
                self.logger.log(32, llm_output)
                raise AgentParsingError("Search query not found in the output.")
            split = llm_output.split('Search:')
            rationale, search_query = (
                split[-2],
                split[-1],
            )
            search_query = search_query.split('Observation:')[0]
            current_step_logs["rationale"] = rationale.split('Thought:')[-1]
            current_step_logs["summarization"] = rationale.split('Thought:')[0].split('Summarization')[-1]
            self.logger.warning(">>>> summarization: ")
            self.logger.log(32, rationale.split('Thought:')[0].split('Summarization')[-1])
            self.logger.warning(">>>> Thought: ")
            self.logger.log(32, rationale.split('Thought:')[-1])
            self.logger.warning(">>>> Search: ")
            self.logger.log(32, search_query)
            search_flag = True
        except:
            try:
                if "Final Answer:" not in llm_output:
                    raise AgentParsingError("Final answer not found in the output.")
                # rationale, final_result = self.extract_action(llm_output=llm_output, split_token="Final Answer:")
                final_result = llm_output.split("Final Answer:")[-1]
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, final_result)
                current_step_logs["final_answer"] = final_result
            except Exception as e:
                self.logger.error(e, exc_info=1)
                self.logger.debug(f"Error in extracting rational and search_query or final answer, trying to parse the whole output. Error trace: {e}")
                rationale, search_query = llm_output, llm_output
                current_step_logs["rationale"] = rationale

        try:
            if search_flag:
                search_res = self.search_tool.forward(search_query) # TODO
                self.logger.warning("Print search result:")
                self.logger.log(32, search_res)
                current_step_logs["observation"] = search_res
        except Exception as e:
            error_msg = f"Error in searching: {e}."
            raise AgentParsingError(error_msg)

        
        # current_step_logs["tool_call"] = {"tool_name": "tavily_search_results", "tool_arguments": search_query}
        
        #########################################################################

        obs = current_step_logs["observation"] if "observation" in current_step_logs else ''
        sm = current_step_logs["summarization"] if "summarization" in current_step_logs else ''
        return current_step_logs, obs, sm # >>
    
