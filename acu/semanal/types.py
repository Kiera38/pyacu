from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

from acu.source import Location


@dataclass
class Type:
    def can_convert(self, to: Type) -> bool:
        return self == to


@dataclass
class NothingType(Type):
    pass


nothing_type = NothingType()


@dataclass
class BoolType(Type):
    def can_convert(self, to: Type) -> bool:
        return super().can_convert(to) or isinstance(to, (IntType, FloatType))


bool_type = BoolType()


@dataclass
class IntType(Type):
    size: int

    def can_convert(self, to: Type) -> bool:
        return super().can_convert(to) or isinstance(to, FloatType) or isinstance(to, IntType) and to.size >= self.size or isinstance(to, BoolType)
    

int_type = IntType(64)


@dataclass
class FloatType(Type):
    size: int

    def can_convert(self, to: Type) -> bool:
        return super().can_convert(to) or isinstance(to, FloatType) and to.size >= self.size or isinstance(to, BoolType)


float_type = FloatType(64)


@dataclass
class FuncType(Type):
    args: list[Type]
    return_type: Type


@dataclass
class ArrayType(Type):
    type: Type
    size: int


@dataclass
class PointerType(Type):
    type: Type


@dataclass
class StructField:
    type: Type
    index: int
    location: Location


@dataclass
class Struct(Type):
    name: str
    fields: dict[str, StructField]
    location: Location

    @cached_property
    def field_list(self):
        return sorted(self.fields.items(), key=lambda f: f[1].index)


@dataclass
class StructType(Type):
    struct: Struct
