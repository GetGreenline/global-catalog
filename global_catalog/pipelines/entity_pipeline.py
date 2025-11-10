class EntityPipeline:
    def __init__(self, repo, matcher, resolver, publisher_fn=None):
        self.repo = repo
        self.matcher = matcher
        self.resolver = resolver
        self.publisher_fn = publisher_fn
        self.pipeline_steps = ['ingest', 'normalize', 'match', 'resolve']
