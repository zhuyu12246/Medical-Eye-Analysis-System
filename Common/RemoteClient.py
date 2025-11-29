# encoding=utf-8
import abc
from abc import abstractmethod


class RemoteClient(abc.ABC):
    @abstractmethod
    def run_by_file(self, file_path: str):
        pass

    def get_url(self):
        """
            如果系统后续拥有多组运算单元，此处添加部分负载均衡算法
            比如使用计数，随机hash等
        :return:
        """
        pass


