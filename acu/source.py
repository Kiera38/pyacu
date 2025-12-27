from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Source:
    name: str
    file: str
    text: str

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines(keepends=False)
    
    def get_lines(self, location: Location, context_lines_before: int = 2, context_lines_after: int = 2) -> str:
        lines = self.lines
        start_line = max(location.line - context_lines_before - 1, 0)
        end_line = min(location.end_line + context_lines_after, len(lines))
        return "\n".join(lines[start_line:end_line])
    

@dataclass(unsafe_hash=True)
class Location:
    line: int
    column: int
    end_line: int
    end_column: int

    @property
    def start(self):
        return (self.line, self.column)
    
    @start.setter
    def start(self, value: tuple[int, int]):
        self.line, self.column = value

    @property
    def end(self):
        return (self.end_line, self.end_column)
    
    @end.setter
    def end(self, value: tuple[int, int]):
        self.end_line, self.end_column = value