import time

class PipelineStep:
    def __init__(self, name):
        self.name = name

    def process(self):
        raise NotImplementedError

    def run(self):
        start_time = time.time()
        result = self.process()
        end_time = time.time()
        logger.info(f"Took {end_time - start_time} seconds.")
        return result

    def get_processed_files(self):
        raise NotImplementedError