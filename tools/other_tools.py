import os

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_search import GoogleSearchAPIWrapper


class Adding(BaseTool):

    name: str = "adding_with_1"
    description: str = (
        "it can return the result of adding a number with 1"
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return int(query.strip()) + 1
    
class python_coding(BaseTool):
    pass

class Time(BaseTool):
    "Query the time"

    name: str = "time"
    description: str = (
        "return current time and date of Beijing timezone, use if you need to know the current time"
    )

    def _run(self, query: str) -> str:
        from datetime import datetime, timedelta, timezone

        current_time = datetime.now()
        beijing_timezone = timezone(timedelta(hours=8))
        beijing_time = current_time.astimezone(beijing_timezone)
        beijing_time_str = beijing_time.strftime('%Y-%m-%d %H:%M:%S')
        return beijing_time_str

