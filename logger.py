import logging


class LogToFrontendHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []

    # If you'd like to set a capacity for the buffer, you can use the following code.
    # def __init__(self, capacity=100, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.buffer = deque(maxlen=capacity)

    def emit(self, record):
        self.buffer.append(self.format(record))

    def clear(self):
        self.buffer.clear()


custom_logger = logging.getLogger("myapp")
custom_logger.setLevel(logging.INFO)

handler = LogToFrontendHandler()
handler.setLevel(logging.INFO)
custom_logger.addHandler(handler)
