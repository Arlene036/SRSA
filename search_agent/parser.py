import re
from typing import Dict, List, Tuple, Optional
from langchain.output_parsers import ListOutputParser
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from pydantic import BaseModel

class AskUserParser(BaseOutputParser):
    def parse(self, text) -> Dict:
        # pattern = r"Answer:\s*(Clear|Unclear)\s*Question:\s*(.*)"
        pattern = r"Clear Score:\s*(.*)\s*Question:\s*(.*)"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            # answer = match.group(1).strip()
            # question = match.group(2).strip()

            # question = None if question == "None" else question
            
            # return {'Answer': answer, 'Question': question}
            clear_score = match.group(1).strip()
            question = match.group(2).strip()

            if int(clear_score) <= 2:
                return {'Answer': 'Unclear', 'Question': question}
            else:
                return {'Answer': 'Clear', 'Question': None}
        else:
            return {'Answer': 'Clear', 'Question': None}


class StrategySuggestionParser(BaseOutputParser):
    def parse(self, text) -> Tuple[str, str]:
        # strategy
        valid_strategies = ["Parallel", "Planning", "Direct"]
        marker = "Strategy:"
        strategy = None

        marker_position = text.find(marker)
        if marker_position != -1:
            # Extract the text following the marker
            start = marker_position + len(marker)
            end = text.find("\n", start)
            strategy = text[start:end].strip()
        else:
            for valid_strategy in valid_strategies:
                if valid_strategy in text:
                    if strategy is not None:
                        return "Multiple strategies found in text: {text}" # raise ValueError
                    strategy = valid_strategy

        if strategy not in valid_strategies:
            strategy = 'Direct'

        # suggestion
        marker = "Suggestions:"
        suggestions = ''
        marker_position = text.find(marker)
        if marker_position != -1:
            # Extract the text following the marker
            start = marker_position + len(marker)
            end = text.find("\n", start)
            suggestions = text[start:end].strip()
        else:
            suggestions = ''
            
        return strategy, suggestions
    
class StrategyParser(BaseOutputParser):
    def parse(self, text) -> Dict:
        valid_strategies = ["Parallel", "Planning", "Direct"]
        marker = "Strategy:"
        strategy = None

        marker_position = text.find(marker)
        if marker_position != -1:
            # Extract the text following the marker
            start = marker_position + len(marker)
            end = text.find("\n", start)
            strategy = text[start:end].strip()
        else:
            # 如果找不到，那就从整个text里面找这三个，如果只有一个，就返回这个，如果有多个，就raise error
            for valid_strategy in valid_strategies:
                if valid_strategy in text:
                    if strategy is not None:
                        raise ValueError(f"Multiple strategies found in text: {text}")
                    strategy = valid_strategy

        # Check if the extracted strategy is valid
        if strategy not in valid_strategies:
            # raise ValueError(f"Invalid strategy: {strategy}")
            return 'Direct'

        return strategy
    
class GeneratedQuestionsSeparatedListOutputParser(ListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "output_parsers", "list"]

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call and extract questions.

        Args:
            text (str): The text output from an LLM call, containing sections and questions.

        Returns:
            List[str]: A list of strings, each representing a found question in the format 'number.question'.
        """
        text = text.lower()
        parts = text.split("generated questions:")

        if len(parts) > 1:
            questions = parts[1].strip().split("\n")
            questions = [q for q in questions if q]
    
        # 提取以数字序列开头的问题
        seq_questions = []
        for q in questions:
            stripped_q = q.strip()
            
            # 检查是否以数字序列开头
            if stripped_q.startswith(tuple(f"{i}." for i in range(1, 10))):
                # 去掉数字序列和后面的空格
                question_text = stripped_q.split(".", 1)[-1].strip()
                
                # 仅在去掉数字序列后问题有实际内容时才加入列表
                if question_text:
                    seq_questions.append(question_text)
            else:
                break  # 如果问题不以数字序列开头，则停止提取
    
        if len(seq_questions) > 5:
            return seq_questions[:5]
        else:
            return seq_questions

    @property
    def _type(self) -> str:
        return "generated questions list"

class RephraseParser(BaseOutputParser):
    def parse(self, text) -> str:
        text = text.lower()
        marker = "rephrased question:"
        rephrased_question = None

        marker_position = text.find(marker)
        if marker_position != -1:
            parts = text.split("rephrased question:")
            if len(parts) > 1:
                rephrased_question = parts[1].strip()
            newline_position = rephrased_question.find('\n')
            if newline_position != -1:
                    rephrased_question = rephrased_question[:newline_position].strip()
        else:
            raise ValueError(f"Rephrased question not found in text: {text}")

        return rephrased_question
