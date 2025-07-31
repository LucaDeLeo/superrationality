# pytest-asyncio Documentation: Async Testing and Fixtures

## Overview

Pytest-asyncio provides support for coroutines as test functions, allowing users to await code inside their tests. The pytest-asyncio plugin simplifies handling event loops, managing async fixtures, and bridges the gap between async programming and thorough testing.

## Installation and Setup

### Basic Installation
```bash
pip install pytest-asyncio
```

### Configuration
To avoid marking every test with `@pytest.mark.asyncio`, enable auto-mode by adding to your `pytest.ini` or `pyproject.toml`:

```ini
# pytest.ini
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

## Writing Async Tests

### Basic Async Test
Tests are marked with `@pytest.mark.asyncio` decorator to execute async code:

```python
import pytest

@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

### With Auto Mode Enabled
If auto mode is enabled, you can omit the decorator:

```python
# No decorator needed with asyncio_mode = "auto"
async def test_some_asyncio_code():
    res = await library.do_something()
    assert b"expected result" == res
```

## Async Fixtures

### Creating Async Fixtures
Use the `@pytest_asyncio.fixture()` marker for async fixtures:

```python
import pytest_asyncio

@pytest_asyncio.fixture
async def async_resource():
    # Setup async resource
    resource = await create_resource()
    yield resource
    # Teardown
    await resource.close()
```

### Using Async Fixtures in Tests
Access async fixtures in your test functions normally:

```python
@pytest.mark.asyncio
async def test_with_async_fixture(async_resource):
    # No need to await the fixture - pytest-asyncio handles it
    result = await async_resource.do_something()
    assert result == expected_value
```

### Context Manager Fixtures
For resources that require proper cleanup (like HTTP/TCP connections):

```python
@pytest_asyncio.fixture
async def async_app_client():
    async with AsyncClient(app=app) as client:
        yield client
```

## Event Loop Management

### Event Loop Scoping
Pytest-asyncio provides one asyncio event loop for each pytest collector. By default, each test runs in its own event loop.

### Sharing Event Loops
Tests can share event loops by specifying the `loop_scope`:

```python
@pytest.mark.asyncio(loop_scope="module")
async def test_uses_shared_loop():
    # This test shares the event loop with other module-scoped tests
    pass

@pytest.mark.asyncio(loop_scope="session")
async def test_session_scoped():
    # Shares loop with all session-scoped tests
    pass
```

Available scopes:
- `"function"` (default): Each test gets its own loop
- `"class"`: Tests in the same class share a loop
- `"module"`: Tests in the same module share a loop
- `"package"`: Tests in the same package share a loop
- `"session"`: All tests share the same loop

## Advanced Patterns

### Async Setup and Teardown
```python
@pytest_asyncio.fixture
async def database_connection():
    # Setup
    conn = await create_connection()
    await conn.execute("CREATE TABLE test (id INT)")
    
    yield conn
    
    # Teardown
    await conn.execute("DROP TABLE test")
    await conn.close()
```

### Parametrized Async Tests
```python
@pytest.mark.asyncio
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
async def test_async_uppercase(input, expected):
    result = await async_uppercase(input)
    assert result == expected
```

### Async Mocking
For mocking async functions, use `AsyncMock`:

```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_with_mock():
    mock_func = AsyncMock(return_value="mocked result")
    result = await mock_func()
    assert result == "mocked result"
```

## Mode Recommendations

### Auto Mode (Recommended)
Auto mode is intended for projects that use asyncio as their only asynchronous programming library. It provides the simplest test and fixture configuration.

### Strict Mode
Use strict mode when you need explicit control over which tests are async:

```ini
[tool.pytest.ini_options]
asyncio_mode = "strict"
```

## Important Notes

1. **Fixture Decorator**: Always use `@pytest_asyncio.fixture` for async fixtures, not `@pytest.fixture`
2. **Test Classes**: Test classes subclassing `unittest` are not supported. Use `unittest.IsolatedAsyncioTestCase` or async frameworks like `asynctest`
3. **Awaiting Fixtures**: You don't need to await fixtures in your tests - pytest-asyncio handles this automatically

## Common Patterns

### Testing Async Context Managers
```python
@pytest.mark.asyncio
async def test_async_context_manager():
    async with MyAsyncContextManager() as manager:
        result = await manager.do_something()
        assert result.status == "success"
```

### Testing Async Generators
```python
@pytest.mark.asyncio
async def test_async_generator():
    results = []
    async for item in async_data_generator():
        results.append(item)
    assert len(results) == expected_count
```

### Timeout Testing
```python
@pytest.mark.asyncio
@pytest.mark.timeout(5)  # 5 second timeout
async def test_with_timeout():
    await long_running_operation()
```

## References
- [Official pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [PyPI Package](https://pypi.org/project/pytest-asyncio/)
- [GitHub Repository](https://github.com/pytest-dev/pytest-asyncio)