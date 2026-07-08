"""
Local task application - background job execution.

All long-running Research Layer jobs (model training, feature generation,
backtest runs, Optuna tuning studies) are dispatched as background tasks so
the FastAPI process returns immediately with a task ID the client can poll.

For local development:
    Tasks are executed in a ThreadPoolExecutor within the same process.
    No external services required.

The API is designed to be task-queue compatible for future scaling needs.
"""
from typing import Any, Callable, Optional
from functools import wraps

from app.workers.task_queue import TaskStatus, get_task_status, submit_task


class LocalTask:
    """
    A task wrapper that provides a task-queue compatible interface.
    
    Usage:
        @local_task(bind=True, name="my_task")
        def my_task(self, arg1, arg2):
            return arg1 + arg2
        
        # Submit task
        task_id = my_task.delay(arg1=1, arg2=2)
        
        # Poll status
        status = my_task.AsyncResult(task_id)
    """
    
    def __init__(self, func: Optional[Callable] = None, *, bind: bool = False, name: Optional[str] = None):
        self._func = func
        self._bind = bind
        self._name = name or (func.__name__ if func else "unknown")
        self._task_id: Optional[str] = None
    
    def __call__(self, *args, **kwargs):
        if self._func is None:
            raise RuntimeError("Task not initialized")
        if self._bind:
            return self._func(self, *args, **kwargs)
        return self._func(*args, **kwargs)
    
    def delay(self, *args, **kwargs) -> str:
        """Submit the task for background execution."""
        func = self._func
        if self._bind and args:
            # If bind=True and first arg is self-like, skip it for submission
            func = lambda *a, **kw: self._func(self, *a, **kw)
        
        task_id = submit_task(
            func,
            *args,
            task_name=self._name,
            **kwargs,
        )
        return task_id
    
    def apply_async(self, args: tuple = (), kwargs: dict = None, **options) -> str:
        """task-queue compatible apply_async method."""
        return self.delay(*args, **(kwargs or {}))
    
    def AsyncResult(self, task_id: str) -> "LocalAsyncResult":
        """Return an AsyncResult for the given task ID."""
        return LocalAsyncResult(task_id)
    
    @property
    def name(self) -> str:
        return self._name


class LocalAsyncResult:
    """
    task-queue compatible AsyncResult for local task queue.
    
    Usage:
        result = MyTask().AsyncResult(task_id)
        print(result.state)  # PENDING, STARTED, SUCCESS, FAILURE
        if result.ready():
            print(result.result)
    """
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self._state: Optional[str] = None
        self._result: Any = None
        self._error: Optional[str] = None
    
    def _refresh(self):
        """Refresh state from task queue."""
        status = get_task_status(self.task_id)
        if status:
            self._state = status.get("status", "PENDING")
            self._result = status.get("result")
            self._error = status.get("error")
    
    @property
    def state(self) -> str:
        self._refresh()
        return self._state or "PENDING"
    
    @property
    def status(self) -> str:
        return self.state
    
    @property
    def result(self) -> Any:
        self._refresh()
        return self._result
    
    @property
    def info(self) -> Any:
        """Alias for result."""
        return self.result
    
    @property
    def error(self) -> Optional[str]:
        self._refresh()
        return self._error
    
    def ready(self) -> bool:
        """Return True if task has completed (success or failure)."""
        self._refresh()
        return self._state in (TaskStatus.SUCCESS.value, TaskStatus.FAILURE.value)
    
    def successful(self) -> bool:
        """Return True if task completed successfully."""
        return self.ready() and self._state == TaskStatus.SUCCESS.value
    
    def failed(self) -> bool:
        """Return True if task failed."""
        return self.ready() and self._state == TaskStatus.FAILURE.value
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """Get the result, blocking until ready."""
        import time
        start = time.time()
        while not self.ready():
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Task {self.task_id} did not complete within {timeout} seconds")
            time.sleep(0.1)
        
        if self.failed():
            raise Exception(self.error or "Task failed")
        return self.result


def local_task(func: Optional[Callable] = None, *, bind: bool = False, name: Optional[str] = None) -> LocalTask:
    """
    Decorator to create a local task.
    
    Usage:
        @local_task
        def my_task(arg1, arg2):
            return arg1 + arg2
        
        @local_task(bind=True, name="custom_name")
        def bound_task(self, arg):
            return arg * 2
        
        # Submit
        task_id = my_task.delay(1, 2)
    """
    def decorator(f: Callable) -> LocalTask:
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return LocalTask(wrapper, bind=bind, name=name or f.__name__)
    
    if func is None:
        return decorator
    return decorator(func)


# Create the local task_app
class LocalTaskApp:
    """
    A task queue app interface for local execution.
    
    Provides task queue compatible interface for background job execution.
    """
    
    def __init__(self, name: str = "research_layer"):
        self.name = name
        self._tasks = {}
    
    def task(self, func: Optional[Callable] = None, *, name: Optional[str] = None, **kwargs) -> LocalTask:
        """Decorator to register a task."""
        def decorator(f: Callable) -> LocalTask:
            task_name = name or f.__name__
            self._tasks[task_name] = LocalTask(f, name=task_name)
            return self._tasks[task_name]
        
        if func is None:
            return decorator
        return decorator(func)
    
    def register_task(self, task: LocalTask) -> None:
        """Register a task manually."""
        self._tasks[task.name] = task
    
    def get_task(self, name: str) -> Optional[LocalTask]:
        """Get a registered task by name."""
        return self._tasks.get(name)


# Create the app instance
task_app = LocalTaskApp("research_layer")


# For backward compatibility with task queue patterns
class AsyncResult:
    """Backward-compatible alias."""
    def __new__(cls, task_id: str, app=None):
        return LocalAsyncResult(task_id)
