"""
Local task queue system for background job execution.

This module provides an in-process background task execution system using:
- concurrent.futures.ThreadPoolExecutor for async task execution
- APScheduler for scheduled jobs
- In-memory task state storage (with optional disk persistence)

Tasks can be submitted from FastAPI endpoints and run in background threads,
with the ability to poll task status and retrieve results.

Usage:
    from app.workers.task_queue import submit_task, get_task_status
    
    # Submit a task
    task_id = submit_task("my_task_func", arg1, arg2)
    
    # Poll for status
    status = get_task_status(task_id)
    
    # Get result when ready
    if status["status"] == "SUCCESS":
        result = status["result"]
"""
from __future__ import annotations

import asyncio
import functools
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional
from threading import Lock

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers import date

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"


@dataclass
class TaskInfo:
    """Information about a submitted task."""
    task_id: str
    name: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class TaskState:
    """Thread-safe in-memory task state storage."""
    
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = Lock()
    
    def create(self, name: str) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = TaskInfo(
                task_id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                created_at=datetime.utcnow(),
            )
        return task_id
    
    def get(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info by ID."""
        with self._lock:
            return self._tasks.get(task_id)
    
    def update(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        result: Any = None,
        error: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        started: bool = False,
        completed: bool = False,
    ) -> Optional[TaskInfo]:
        """Update task state."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            
            if status:
                task.status = status
            if started and task.started_at is None:
                task.started_at = datetime.utcnow()
            if completed:
                task.completed_at = datetime.utcnow()
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if meta:
                task.meta.update(meta)
            
            return task
    
    def list_all(self) -> list[TaskInfo]:
        """List all tasks."""
        with self._lock:
            return list(self._tasks.values())


# Global task state
_task_state = TaskState()

# Thread pool for background execution
_thread_pool: Optional[ThreadPoolExecutor] = None

# APScheduler for scheduled jobs
_scheduler: Optional[BackgroundScheduler] = None


def _get_thread_pool() -> ThreadPoolExecutor:
    """Get or create the thread pool executor."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="task_worker_")
    return _thread_pool


def _get_scheduler() -> BackgroundScheduler:
    """Get or create the APScheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone="UTC")
        _scheduler.start()
    return _scheduler


def submit_task(func: Callable, *args, task_name: Optional[str] = None, **kwargs) -> str:
    """
    Submit a function to run in the background thread pool.
    
    Args:
        func: The function to execute
        *args: Positional arguments to pass to the function
        task_name: Optional name for the task (defaults to function name)
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        task_id: A string ID that can be used to poll for status/results
    """
    task_id = _task_state.create(task_name or func.__name__)
    executor = _get_thread_pool()
    
    # Update status to started
    _task_state.update(task_id, status=TaskStatus.STARTED, started=True)
    
    def _run():
        """Wrapper to run the task and update state."""
        try:
            # For async functions, run them in an event loop
            if asyncio.iscoroutinefunction(func):
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(func(*args, **kwargs))
                finally:
                    loop.close()
            else:
                result = func(*args, **kwargs)
            
            _task_state.update(
                task_id,
                status=TaskStatus.SUCCESS,
                result=result,
                completed=True,
            )
        except Exception as exc:
            logger.exception(f"Task {task_id} failed: {exc}")
            _task_state.update(
                task_id,
                status=TaskStatus.FAILURE,
                error=str(exc),
                completed=True,
            )
    
    executor.submit(_run)
    return task_id


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a task.
    
    Args:
        task_id: The task ID returned by submit_task
    
    Returns:
        Dict with status, result, error, and metadata, or None if task not found
    """
    task = _task_state.get(task_id)
    if task is None:
        return None
    
    result = {
        "task_id": task.task_id,
        "name": task.name,
        "status": task.status.value,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }
    
    if task.started_at:
        result["started_at"] = task.started_at.isoformat()
    if task.completed_at:
        result["completed_at"] = task.completed_at.isoformat()
    if task.meta:
        result["meta"] = task.meta
    if task.error:
        result["error"] = task.error
    
    # Include result only if completed successfully
    if task.status == TaskStatus.SUCCESS and task.result is not None:
        result["result"] = task.result
    
    return result


def schedule_task(
    func: Callable,
    run_date: datetime,
    *args,
    task_name: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Schedule a function to run at a specific time.
    
    Args:
        func: The function to execute
        run_date: When to run the function
        *args: Positional arguments to pass to the function
        task_name: Optional name for the task
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        task_id: A string ID for the scheduled task
    """
    task_id = submit_task(func, *args, task_name=task_name, **kwargs)
    
    scheduler = _get_scheduler()
    scheduler.add_job(
        _execute_task,
        trigger=date.DateRunTime(run_date=run_date),
        args=[task_id, func, args, kwargs],
        id=task_id,
        name=task_name or func.__name__,
    )
    
    return task_id


def _execute_task(task_id: str, func: Callable, args: tuple, kwargs: dict):
    """Internal function to execute a scheduled task."""
    try:
        if asyncio.iscoroutinefunction(func):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(func(*args, **kwargs))
            finally:
                loop.close()
        else:
            result = func(*args, **kwargs)
        
        _task_state.update(
            task_id,
            status=TaskStatus.SUCCESS,
            result=result,
            completed=True,
        )
    except Exception as exc:
        logger.exception(f"Scheduled task {task_id} failed: {exc}")
        _task_state.update(
            task_id,
            status=TaskStatus.FAILURE,
            error=str(exc),
            completed=True,
        )


def shutdown():
    """Shutdown the task queue gracefully."""
    global _thread_pool, _scheduler
    
    if _scheduler:
        _scheduler.shutdown(wait=True)
        _scheduler = None
    
    if _thread_pool:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None


def is_ready() -> bool:
    """Check if the task queue is ready."""
    return _get_thread_pool() is not None
