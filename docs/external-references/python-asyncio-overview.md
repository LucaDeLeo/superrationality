# Python Asyncio Documentation

## Overview

asyncio is a library to write **concurrent** code using the **async/await** syntax.

asyncio is used as a foundation for multiple Python asynchronous frameworks that provide high-performance network and web-servers, database connection libraries, distributed task queues, etc.

asyncio is often a perfect fit for IO-bound and high-level **structured** network code.

## Hello World Example

```python
import asyncio

async def main():
    print('Hello ...')
    await asyncio.sleep(1)
    print('... World!')

asyncio.run(main())
```

## High-level APIs

asyncio provides a set of **high-level** APIs to:

- [run Python coroutines](https://docs.python.org/3/library/asyncio-task.html#coroutine) concurrently and have full control over their execution
- perform [network IO and IPC](https://docs.python.org/3/library/asyncio-stream.html#asyncio-streams)
- control [subprocesses](https://docs.python.org/3/library/asyncio-subprocess.html#asyncio-subprocess)
- distribute tasks via [queues](https://docs.python.org/3/library/asyncio-queue.html#asyncio-queues)
- [synchronize](https://docs.python.org/3/library/asyncio-sync.html#asyncio-sync) concurrent code

## Low-level APIs

Additionally, there are **low-level** APIs for _library and framework developers_ to:

- create and manage [event loops](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio-event-loop), which provide asynchronous APIs for networking, running subprocesses, handling OS signals, etc
- implement efficient protocols using [transports](https://docs.python.org/3/library/asyncio-protocol.html#asyncio-transports-protocols)
- [bridge](https://docs.python.org/3/library/asyncio-future.html#asyncio-futures) callback-based libraries and code with async/await syntax

## asyncio REPL

You can experiment with an `asyncio` concurrent context in the REPL:

```bash
$ python -m asyncio
asyncio REPL ...
Use "await" directly instead of "asyncio.run()".
Type "help", "copyright", "credits" or "license" for more information.
>>> import asyncio
>>> await asyncio.sleep(10, result='hello')
'hello'
```

## Reference Documentation

### High-level APIs
- [Runners](https://docs.python.org/3/library/asyncio-runner.html)
- [Coroutines and Tasks](https://docs.python.org/3/library/asyncio-task.html)
- [Streams](https://docs.python.org/3/library/asyncio-stream.html)
- [Synchronization Primitives](https://docs.python.org/3/library/asyncio-sync.html)
- [Subprocesses](https://docs.python.org/3/library/asyncio-subprocess.html)
- [Queues](https://docs.python.org/3/library/asyncio-queue.html)
- [Exceptions](https://docs.python.org/3/library/asyncio-exceptions.html)

### Low-level APIs
- [Event Loop](https://docs.python.org/3/library/asyncio-eventloop.html)
- [Futures](https://docs.python.org/3/library/asyncio-future.html)
- [Transports and Protocols](https://docs.python.org/3/library/asyncio-protocol.html)
- [Policies](https://docs.python.org/3/library/asyncio-policy.html)
- [Platform Support](https://docs.python.org/3/library/asyncio-platforms.html)
- [Extending](https://docs.python.org/3/library/asyncio-extending.html)

### Guides and Tutorials
- [High-level API Index](https://docs.python.org/3/library/asyncio-api-index.html)
- [Low-level API Index](https://docs.python.org/3/library/asyncio-llapi-index.html)
- [Developing with asyncio](https://docs.python.org/3/library/asyncio-dev.html)

**Note**: The source code for asyncio can be found in [Lib/asyncio/](https://github.com/python/cpython/tree/3.13/Lib/asyncio/).

**Availability**: not WASI. This module does not work or is not available on WebAssembly.