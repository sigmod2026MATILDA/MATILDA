import psutil
import asyncio
import time
import logging
from datetime import datetime

class ResourceMonitor:
    """Monitors memory usage and timeout constraints."""

    def __init__(self, memory_threshold: int, timeout: int):
        self.memory_threshold = memory_threshold
        self.timeout = timeout
        self.start_time = time.time()
        self.process = psutil.Process()
        self.logger = logging.getLogger()

    async def monitor(self):
        """Continuously monitors memory and execution time."""
        while True:
            elapsed_time = time.time() - self.start_time
            memory_usage = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                memory_usage += child.memory_info().rss

            memory_usage_gb = memory_usage / (1024 ** 3)

            self.logger.info(f"Total Memory Usage: {memory_usage_gb:.2f} GB")

            if memory_usage_gb > (self.memory_threshold / (1024 ** 3)):
                self.logger.error(f"Memory usage exceeded: {memory_usage_gb:.2f} GB")
                raise MemoryError("Memory usage exceeded threshold.")

            if elapsed_time > self.timeout:
                self.logger.error("Execution timeout exceeded.")
                raise TimeoutError("Execution timeout exceeded.")

            await asyncio.sleep(5)
