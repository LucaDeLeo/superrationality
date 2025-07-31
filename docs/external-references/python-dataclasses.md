# Python Dataclasses Documentation

## Overview

The `dataclasses` module provides a decorator and functions for automatically adding generated special methods such as `__init__()` and `__repr__()` to user-defined classes. It was originally described in [PEP 557](https://peps.python.org/pep-0557/).

**Source code:** [Lib/dataclasses.py](https://github.com/python/cpython/tree/3.13/Lib/dataclasses.py)

## Basic Usage

```python
from dataclasses import dataclass

@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
```

This will automatically add an `__init__()` method that looks like:

```python
def __init__(self, name: str, unit_price: float, quantity_on_hand: int = 0):
    self.name = name
    self.unit_price = unit_price
    self.quantity_on_hand = quantity_on_hand
```

## The @dataclass Decorator

### Basic Syntax

```python
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, 
           frozen=False, match_args=True, kw_only=False, slots=False, 
           weakref_slot=False)
class ClassName:
    ...
```

### Parameters

- **init**: If true (default), generates `__init__()` method
- **repr**: If true (default), generates `__repr__()` method
- **eq**: If true (default), generates `__eq__()` method
- **order**: If true (default False), generates comparison methods (`__lt__()`, `__le__()`, `__gt__()`, `__ge__()`)
- **unsafe_hash**: If true, forces creation of `__hash__()` method
- **frozen**: If true (default False), emulates read-only frozen instances
- **match_args**: If true (default), creates `__match_args__` tuple
- **kw_only**: If true (default False), all fields become keyword-only
- **slots**: If true (default False), generates `__slots__` attribute
- **weakref_slot**: If true (default False), adds `__weakref__` slot

## Field Specifications

### Default Values

```python
@dataclass
class C:
    a: int       # 'a' has no default value
    b: int = 0   # assign a default value for 'b'
```

### The field() Function

For advanced field configuration:

```python
from dataclasses import dataclass, field

@dataclass
class C:
    mylist: list[int] = field(default_factory=list)
    hidden: int = field(repr=False, default=0)
    compare_only: str = field(compare=True, hash=False)
```

#### field() Parameters

- **default**: Default value for the field
- **default_factory**: Zero-argument callable for mutable defaults
- **init**: Include in `__init__()` (default True)
- **repr**: Include in `__repr__()` (default True)
- **hash**: Include in `__hash__()` (default None)
- **compare**: Include in comparison methods (default True)
- **metadata**: Mapping for third-party extensions
- **kw_only**: Mark as keyword-only (default False)

## Special Field Types

### ClassVar - Class Variables

```python
from typing import ClassVar

@dataclass
class C:
    instance_var: int
    class_var: ClassVar[int] = 0  # Not included in __init__
```

### InitVar - Init-Only Variables

```python
from dataclasses import InitVar

@dataclass
class C:
    i: int
    j: int = None
    database: InitVar[DatabaseType] = None

    def __post_init__(self, database):
        if self.j is None and database is not None:
            self.j = database.lookup('j')
```

### KW_ONLY - Keyword-Only Fields

```python
from dataclasses import KW_ONLY

@dataclass
class Point:
    x: float
    _: KW_ONLY
    y: float
    z: float

p = Point(0, y=1.5, z=2.0)  # y and z must be passed as keywords
```

## Post-Init Processing

```python
@dataclass
class C:
    a: float
    b: float
    c: float = field(init=False)

    def __post_init__(self):
        self.c = self.a + self.b
```

## Inheritance

```python
@dataclass
class Base:
    x: Any = 15.0
    y: int = 0

@dataclass
class C(Base):
    z: int = 10
    x: int = 15  # Override parent's field
```

## Frozen Instances

Create immutable-like objects:

```python
@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
# p.x = 3.0  # Raises FrozenInstanceError
```

## Utility Functions

### fields()

Get field information:

```python
from dataclasses import fields

@dataclass
class C:
    x: int
    y: int

for field in fields(C):
    print(field.name, field.type)
```

### asdict() and astuple()

Convert to dict or tuple:

```python
from dataclasses import asdict, astuple

@dataclass
class Point:
    x: int
    y: int

p = Point(10, 20)
print(asdict(p))   # {'x': 10, 'y': 20}
print(astuple(p))  # (10, 20)
```

### replace()

Create a new instance with modified fields:

```python
from dataclasses import replace

@dataclass
class Point:
    x: int
    y: int

p1 = Point(10, 20)
p2 = replace(p1, x=30)  # Point(x=30, y=20)
```

### is_dataclass()

Check if object is a dataclass:

```python
from dataclasses import is_dataclass

@dataclass
class C:
    x: int

print(is_dataclass(C))     # True
print(is_dataclass(C()))   # True
```

### make_dataclass()

Dynamically create dataclasses:

```python
from dataclasses import make_dataclass

C = make_dataclass('C',
                   [('x', int),
                    'y',
                    ('z', int, field(default=5))],
                   namespace={'add_one': lambda self: self.x + 1})
```

## Best Practices

### Mutable Default Values

Avoid mutable defaults directly:

```python
# Bad - will raise ValueError
@dataclass
class D:
    x: list = []  # Don't do this!

# Good - use default_factory
@dataclass
class D:
    x: list = field(default_factory=list)
```

### Descriptor-Typed Fields

Dataclasses work with descriptors:

```python
class IntConversionDescriptor:
    def __init__(self, *, default):
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default
        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        setattr(obj, self._name, int(value))

@dataclass
class InventoryItem:
    quantity_on_hand: IntConversionDescriptor = IntConversionDescriptor(default=100)
```

## Common Patterns

### Configuration Classes

```python
@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    timeout: float = 30.0
```

### Nested Dataclasses

```python
@dataclass
class Address:
    street: str
    city: str
    zip_code: str

@dataclass
class Person:
    name: str
    age: int
    address: Address
```

### With Type Validation

```python
@dataclass
class ValidatedPoint:
    x: float
    y: float

    def __post_init__(self):
        if not isinstance(self.x, (int, float)):
            raise TypeError(f"x must be numeric, got {type(self.x)}")
        if not isinstance(self.y, (int, float)):
            raise TypeError(f"y must be numeric, got {type(self.y)}")
```

## Version History

- Added in Python 3.7
- `match_args` added in 3.10
- `kw_only` added in 3.10
- `slots` added in 3.10
- `weakref_slot` added in 3.11