# encoding=utf-8
import threading
from enum import IntEnum


class TASK_STATUS(IntEnum):
    STATUS_CREATED = 0
    STATUS_RUNNING = 100
    STATUS_OK = 200
    STATUS_ERROR = 500


class TaskManager:
    def __init__(self, task_key: str, thread_obj: threading.Thread):
        self.task_status = TASK_STATUS.STATUS_CREATED
        self.msg_list = []
        self.error_msg = ""
        self.progress = 0
        self.task_key = task_key
        self.thread_obj = thread_obj

    def task_start(self):
        self.thread_obj.start()

    def get_task_status(self):
        return self.task_status

    def get_next_running_msg(self):
        if len(self.msg_list) == 0:
            return ""

        return self.msg_list.pop(0)

    def get_error_msg(self):
        return not (self.error_msg == ""), self.error_msg

    def get_progress(self):
        return self.progress

    def set_running_state(self, state: TASK_STATUS):
        self.task_status = state

    def set_normal_msg(self, msg: str):
        self.msg_list.append(msg)

    def set_error_msg(self, msg: str):
        self.error_msg = msg

    def set_progress(self, rate: float):
        self.progress = rate
