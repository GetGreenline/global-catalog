import logging

class Logger:
    def __init__(self, name=None, level=logging.INFO):
        self.logger = logging.getLogger(name or __name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(level)

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)

# Usage example:
# logger = Logger(__name__)
# logger.info('This is an info message')
# logger.error('This is an error message')
# logger.log('Custom log level', level=logging.WARNING)
