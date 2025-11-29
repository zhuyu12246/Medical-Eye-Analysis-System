# encoding=utf-8

"""TaskObjAsync.py

    如果后续出现多组运算单元，对TaskObj进行重构
    主要为在threading中引入async

    主要逻辑为主线程等待协程完成, 类似如下实现

        import asyncio

        # 定义第一个async函数
        async def async_function_1():
            await asyncio.sleep(2)  # 模拟耗时操作
            return "Async result 1"

        # 定义第二个async函数
        async def async_function_2():
            await asyncio.sleep(3)  # 模拟耗时操作
            raise ValueError("Oops! An error occurred in async_function_2.")

        # 定义第三个async函数
        async def async_function_3():
            await asyncio.sleep(1)  # 模拟耗时操作
            return "Async result 3"

        # 创建事件循环
        loop = asyncio.new_event_loop()

        # 在事件循环中运行多个异步函数，并等待结果（包括异常）
        results = loop.run_until_complete(asyncio.gather(
            async_function_1(),
            async_function_2(),
            async_function_3(),
            return_exceptions=True  # 设置return_exceptions参数为True
        ))

        # 打印结果
        for result in results:
            if isinstance(result, Exception):
                print("Exception:", result)
            else:
                print("Result:", result)

        # 关闭事件循环
        loop.close()
"""