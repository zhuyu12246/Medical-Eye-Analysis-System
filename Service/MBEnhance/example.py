import cv2
import numpy

from Service.MBEnhance.client import MBEnhanceClient
from Common.RemoteClient import RemoteClient

client: RemoteClient = MBEnhanceClient()

res = client.run_by_file("test.jpg")
if res is None:
    print("远程访问链接失败或图像解析异常")
else:
    status_code, content = res
    if status_code != 200:
        print("远端执行异常")
    else:
        image = cv2.imdecode(numpy.frombuffer(content, numpy.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("out.png", image)
