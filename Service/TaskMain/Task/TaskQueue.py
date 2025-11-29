# encoding=utf-8
import threading
from Service.TaskMain.Task.TaskManager import TaskManager


class TaskQueue:
    _inst = None
    _lock = threading.Lock()

    def __init__(self, max_running_instance: int = 1):
        self.task_dict = {}  # task-manager
        self.task_list = []  # task id
        self.task_running_list = []  # running task
        self.task_finish_list = []  # finished task
        self.max_running_instance = max_running_instance  # running limit

    def task_in(self, task_key: str, task_manager: TaskManager):
        self.task_dict[task_key] = task_manager
        self.task_list.append(task_key)

    def task_out(self):
        if len(self.task_list) == 0:
            return None

        if len(self.task_running_list) >= self.max_running_instance:
            return None

        next_task = self.task_list.pop(0)
        self.task_running_list.append(next_task)
        return next_task

    def task_query(self, task_key: str):
        if task_key in self.task_dict:
            return self.task_dict[task_key]
        return None

    def task_done(self, task_key: str):
        self.task_running_list.remove(task_key)
        self.task_finish_list.append(task_key)

    def task_clean(self, task_key: str):
        del self.task_dict[task_key]
        self.task_finish_list.remove(task_key)

    def task_done_query(self):
        if len(self.task_finish_list) == 0:
            return None
        else:
            return self.task_finish_list[0]

    def __new__(cls, *args, **kwargs):
        TaskQueue._lock.acquire()
        if cls._inst is None:
            cls._inst = super(TaskQueue, cls).__new__(cls)
        TaskQueue._lock.release()
        return cls._inst