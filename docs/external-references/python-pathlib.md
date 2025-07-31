# Python pathlib Documentation - Object-Oriented Filesystem Paths

## Overview

The `pathlib` module offers classes representing filesystem paths with semantics appropriate for different operating systems. Introduced in Python 3.4, it provides an elegant object-oriented approach to handling file system paths, ensuring platform-agnostic behavior.

**Source code:** [Lib/pathlib/](https://github.com/python/cpython/tree/3.13/Lib/pathlib/)

## Key Concepts

### Path Classes Hierarchy

- **Pure paths**: Provide purely computational operations without I/O
  - `PurePath`: Base class for pure paths
  - `PurePosixPath`: Pure path for POSIX systems
  - `PureWindowsPath`: Pure path for Windows

- **Concrete paths**: Inherit from pure paths and provide I/O operations
  - `Path`: Main class (what you'll use most often)
  - `PosixPath`: Concrete path for POSIX systems
  - `WindowsPath`: Concrete path for Windows

## Basic Usage

### Importing and Creating Paths

```python
from pathlib import Path

# Create a Path object
p = Path('.')  # Current directory
p = Path('/usr/bin/python3')  # Absolute path
p = Path('~/documents').expanduser()  # Expand home directory
```

### Path Operations

```python
# Path components
p = Path('/home/user/documents/file.txt')
print(p.name)       # 'file.txt'
print(p.stem)       # 'file'
print(p.suffix)     # '.txt'
print(p.parent)     # Path('/home/user/documents')
print(p.parents[0]) # Path('/home/user/documents')
print(p.parents[1]) # Path('/home/user')
```

## Common Operations

### Joining Paths

```python
# Using the / operator
base = Path('/home/user')
full_path = base / 'documents' / 'file.txt'
# Result: Path('/home/user/documents/file.txt')

# Using joinpath()
full_path = base.joinpath('documents', 'file.txt')
```

### Checking Path Properties

```python
p = Path('myfile.txt')

# Existence and type checks
p.exists()      # Check if path exists
p.is_file()     # Check if it's a file
p.is_dir()      # Check if it's a directory
p.is_symlink()  # Check if it's a symbolic link

# Path properties
p.is_absolute() # Check if path is absolute
p.is_relative_to('/home')  # Check if relative to another path
```

### Reading and Writing Files

```python
# Reading text
content = Path('file.txt').read_text(encoding='utf-8')

# Writing text
Path('file.txt').write_text('Hello, World!', encoding='utf-8')

# Reading bytes
data = Path('image.png').read_bytes()

# Writing bytes
Path('output.bin').write_bytes(data)
```

### Directory Operations

```python
# List directory contents
p = Path('.')
for child in p.iterdir():
    print(child)

# Recursive glob patterns
for py_file in p.glob('**/*.py'):
    print(py_file)

# Using rglob (recursive glob)
for txt_file in p.rglob('*.txt'):
    print(txt_file)
```

### Creating and Removing

```python
# Create directories
Path('new_dir').mkdir()
Path('nested/dir').mkdir(parents=True, exist_ok=True)

# Create file
Path('new_file.txt').touch()

# Remove file
Path('file.txt').unlink()

# Remove empty directory
Path('empty_dir').rmdir()
```

### Path Resolution

```python
p = Path('.')

# Get absolute path
abs_path = p.absolute()

# Resolve symlinks and relative paths
resolved = p.resolve()

# Get relative path
rel = Path('/home/user/file.txt').relative_to('/home')
# Result: Path('user/file.txt')
```

## Advanced Features

### Pattern Matching

```python
# Match against glob pattern
p = Path('myfile.py')
p.match('*.py')  # True

# Case-sensitive matching
p.match('*.PY', case_sensitive=False)  # True on all platforms
```

### File Information

```python
p = Path('myfile.txt')

# Get file stats
stat = p.stat()
print(stat.st_size)  # File size in bytes
print(stat.st_mtime) # Modification time

# Owner and group (Unix)
print(p.owner())
print(p.group())
```

### Working with Symbolic Links

```python
# Create symbolic link
link = Path('mylink')
link.symlink_to('target.txt')

# Read link target
target = link.readlink()
```

## Platform-Specific Behavior

```python
# Automatic platform detection
p = Path('C:\\Users' if os.name == 'nt' else '/home')

# Force specific path type
from pathlib import PureWindowsPath, PurePosixPath

# Manipulate Windows paths on Unix (or vice versa)
win_path = PureWindowsPath('C:\\Users\\file.txt')
posix_path = PurePosixPath('/home/user/file.txt')
```

## Best Practices

### Use Path Instead of os.path

```python
# Old way (os.path)
import os
path = os.path.join('home', 'user', 'file.txt')
if os.path.exists(path):
    with open(path, 'r') as f:
        content = f.read()

# New way (pathlib)
from pathlib import Path
path = Path('home') / 'user' / 'file.txt'
if path.exists():
    content = path.read_text()
```

### Exception Handling

```python
from pathlib import Path

try:
    content = Path('file.txt').read_text()
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("Permission denied")
```

### Working with Configuration Files

```python
# Common pattern for config files
config_dir = Path.home() / '.myapp'
config_dir.mkdir(exist_ok=True)
config_file = config_dir / 'config.json'

# Save config
import json
config = {'setting': 'value'}
config_file.write_text(json.dumps(config))

# Load config
if config_file.exists():
    config = json.loads(config_file.read_text())
```

## Comparison with os.path

| os.path | pathlib |
|---------|---------|
| `os.path.join(a, b)` | `Path(a) / b` |
| `os.path.exists(path)` | `Path(path).exists()` |
| `os.path.isfile(path)` | `Path(path).is_file()` |
| `os.path.isdir(path)` | `Path(path).is_dir()` |
| `os.path.basename(path)` | `Path(path).name` |
| `os.path.dirname(path)` | `Path(path).parent` |
| `os.path.abspath(path)` | `Path(path).absolute()` |

## Common Patterns

### Find All Python Files

```python
# In current directory and subdirectories
python_files = list(Path('.').rglob('*.py'))

# Only in current directory
python_files = list(Path('.').glob('*.py'))
```

### Clean Directory

```python
def clean_directory(dir_path, pattern='*.tmp'):
    """Remove all files matching pattern"""
    path = Path(dir_path)
    for file in path.glob(pattern):
        file.unlink()
```

### Safe File Operations

```python
def safe_write(filepath, content):
    """Write to file safely with backup"""
    path = Path(filepath)
    if path.exists():
        # Create backup
        backup = path.with_suffix(path.suffix + '.bak')
        backup.write_bytes(path.read_bytes())
    
    # Write new content
    path.write_text(content)
```

## Performance Notes

- pathlib is written in pure Python and may be slower than os.path for some operations
- For most use cases, the performance difference is negligible
- The improved readability and functionality often outweigh minor performance costs

## References

- [Official Python Documentation](https://docs.python.org/3/library/pathlib.html)
- [PEP 428](https://peps.python.org/pep-0428/) - The pathlib module proposal