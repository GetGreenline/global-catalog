from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from global_catalog.common.logger import Logger


@dataclass
class EntityPipelineContext:
    raw_data: Optional[Dict[str, Any]] = None
    normalized: Any = None
    match_results: Optional[Dict[str, Any]] = None
    resolution: Any = None
    publish_result: Any = None


class EntityPipeline(ABC):

    def __init__(self, repo, matcher, resolver, publisher_fn=None):
        self.repo = repo
        self.matcher = matcher
        self.resolver = resolver
        self.publisher_fn = publisher_fn
        self.logger = Logger(self.__class__.__name__)
        self.pipeline_steps: tuple[str, ...] = ("ingest", "normalize", "match", "resolve")
        self._context = EntityPipelineContext()
        self._last_run: Dict[str, Any] = {}


    @abstractmethod
    def ingest(self) -> Dict[str, Any]:
        # load data from source (repo, files, etc)
        pass

    @abstractmethod
    def normalize(self, raw_data: Dict[str, Any]) -> Any:
        # clean the data before matching
        pass

    @abstractmethod
    def match(self, normalized_data: Any, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        # match pairs and compare similarity
        pass

    @abstractmethod
    def resolve(
        self,
        match_results: Dict[str, Any],
        raw_data: Dict[str, Any],
        normalized_data: Any,
    ) -> Any:
        pass


    def publish(self, context: EntityPipelineContext) -> Optional[Any]:

        if not callable(self.publisher_fn):
            return None
        return self.publisher_fn(context=context)

    def reset_context(self) -> None:
        self._context = EntityPipelineContext()

    def run(self) -> Dict[str, Any]:

        self.reset_context()
        ctx = self._context
        self.logger.info("PIPELINE: ingest")
        ctx.raw_data = raw_data = self.ingest()
        self.logger.info("PIPELINE: normalize")
        ctx.normalized = normalized = self.normalize(raw_data)
        self.logger.info("PIPELINE: match")
        ctx.match_results = match_results = self.match(normalized, raw_data)
        self.logger.info("PIPELINE: resolve")
        ctx.resolution = resolution = self.resolve(match_results, raw_data, normalized)

        ctx.publish_result = self.publish(ctx)

        self._last_run = asdict(ctx)
        self.logger.info("PIPELINE: done")
        return self._last_run

    @property
    def last_run(self) -> Dict[str, Any]:
        return self._last_run
