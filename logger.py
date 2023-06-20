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


custom_logger = logging.getLogger("value-cumulation")
custom_logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Handler for buffering logs and returning them in the /logs route
handler = LogToFrontendHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
custom_logger.addHandler(handler)

# Handler for printing logs to the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
custom_logger.addHandler(stream_handler)
