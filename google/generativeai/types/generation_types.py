from __future__ import annotations

import collections
from collections.abc import Iterable, AsyncIterable
import dataclasses
import itertools
import textwrap
from typing import TypedDict, Union

from google.ai import generativelanguage as glm

__all__ = [
    "BlockedPromptException",
    "StopCandidateException",
    "IncompleteIterationError",
    "BrokenResponseError",
    "GenerationConfigDict",
    "GenerationConfigType",
    "GenerationConfig",
    "GenerateContentResponse",
]


class BlockedPromptException(Exception):
    pass


class StopCandidateException(Exception):
    pass


class IncompleteIterationError(Exception):
    pass


class BrokenResponseError(Exception):
    pass


class GenerationConfigDict(TypedDict):
    # TODO(markdaoust): Python 3.11+ use `NotRequired`, ref: https://peps.python.org/pep-0655/
    candidate_count: int
    stop_sequences: Iterable[str]
    max_output_tokens: int
    temperature: float


@dataclasses.dataclass
class GenerationConfig:
    candidate_count: int | None = 0
    stop_sequences: Iterable[str] | None = dataclasses.field(default_factory=list)
    max_output_tokens: int | None = 0
    temperature: float | None = 0.0


GenerationConfigType = Union[glm.GenerationConfig, GenerationConfigDict, GenerationConfig]


def to_generation_config_dict(generation_config: GenerationConfigType):
    if generation_config is None:
        return {}
    elif isinstance(generation_config, glm.GenerationConfig):
        return type(generation_config).to_dict(generation_config)
    elif isinstance(generation_config, GenerationConfig):
        return dataclasses.asdict(generation_config)
    elif hasattr(generation_config, "keys"):
        return dict(generation_config)
    else:
        raise TypeError(
            "Did not understand `generation_config`, expected a `dict` or"
            f" `GenerationConfig`\nGot type: {type(generation_config)}\nValue:"
            f" {generation_config}"
        )


def _join_citation_metadatas(
    citation_metadatas: Iterable[glm.CitationMetadata],
):
    citation_sources = []
    for citation_metadata in citation_metadatas:
        citation_sources.extend(citation_metadata.citation_sources)

    return glm.CitationMetadata(citation_sources=citation_sources)


def _join_safety_ratings_lists(
    safety_ratings_lists: Iterable[list[glm.SafetyRating]],
):
    ratings = {}
    blocked = collections.defaultdict(list)

    for safety_ratings_list in safety_ratings_lists:
        for rating in safety_ratings_list:
            ratings[rating.category] = rating.probability
            blocked[rating.category].append(rating.blocked)

    blocked = {category: any(blocked) for category, blocked in blocked.items()}

    safety_list = []
    for (category, probability), blocked in zip(ratings.items(), blocked.values()):
        safety_list.append(
            glm.SafetyRating(category=category, probability=probability, blocked=blocked)
        )

    return safety_list


def _join_contents(contents: Iterable[glm.Content]):
    contents = tuple(contents)
    role = contents[0].role
    assert all(c.role == role for c in contents)

    parts = []
    for content in contents:
        parts.extend(content.parts)

    merged_parts = [parts.pop(0)]
    for part in parts:
        if not merged_parts[-1].text:
            merged_parts.append(part)
            continue

        if not part.text:
            merged_parts.append(part)
            continue

        merged_part = glm.Part(merged_parts[-1])
        merged_part.text += part.text
        merged_parts[-1] = merged_part

    return glm.Content(
        role=role,
        parts=merged_parts,
    )


def _join_candidates(candidates: Iterable[glm.Candidate]):
    candidates = tuple(candidates)

    index = candidates[0].index
    assert all(c.index == index for c in candidates)

    return glm.Candidate(
        index=index,
        content=_join_contents([c.content for c in candidates]),
        finish_reason=candidates[-1].finish_reason,
        safety_ratings=_join_safety_ratings_lists([c.safety_ratings for c in candidates]),
        citation_metadata=_join_citation_metadatas([c.citation_metadata for c in candidates]),
    )


def _join_candidate_lists(candidate_lists: Iterable[list[glm.Candidate]]):
    # Assuming that is a candidate ends, it is no longer returnecd in the list of
    # candidates and that's why candidates have an index
    candidates = collections.defaultdict(list)
    for candidate_list in candidate_lists:
        for candidate in candidate_list:
            candidates[candidate.index].append(candidate)

    new_candidates = []
    for index, candidate_parts in sorted(candidates.items()):
        new_candidates.append(_join_candidates(candidate_parts))

    return new_candidates


def _join_prompt_feedbacks(
    prompt_feedbacks: Iterable[glm.GenerateContentResponse.PromptFeedback],
):
    # Always return the first prompt feedback.
    return next(iter(prompt_feedbacks))


def _join_chunks(chunks: Iterable[glm.GenerateContentResponse]):
    return glm.GenerateContentResponse(
        candidates=_join_candidate_lists(c.candidates for c in chunks),
        prompt_feedback=_join_prompt_feedbacks(c.prompt_feedback for c in chunks),
    )


_INCOMPLETE_ITERATION_MESSAGE = """\
Please let the response complete iteration before accessing the final accumulated
attributes (or call `response.resolve()`)"""


class BaseGenerateContentResponse:
    def __init__(
        self,
        done: bool,
        iterator: None
        | Iterable[glm.GenerateContentResponse]
        | AsyncIterable[glm.GenerateContentResponse],
        result: glm.GenerateContentResponse,
        chunks: Iterable[glm.GenerateContentResponse],
    ):
        self._done = done
        self._iterator = iterator
        self._result = result
        self._chunks = list(chunks)
        if result.prompt_feedback.block_reason:
            self._error = BlockedPromptException(result)
        else:
            self._error = None

    @property
    def candidates(self):
        if not self._done:
            raise IncompleteIterationError(_INCOMPLETE_ITERATION_MESSAGE)
        return self._result.candidates

    @property
    def parts(self):
        candidates = self.candidates
        if not candidates:
            raise ValueError(
                "The `response.parts` quick accessor only works for a single candidate, "
                "but none were returned. Check the `response.prompt_feedback` to see if the prompt was blocked."
            )
        if len(candidates) > 1:
            raise ValueError(
                "The `response.parts` quick accessor only works with a "
                "single candidate. With multiple candidates use "
                "result.candidates[index].text"
            )
        parts = candidates[0].content.parts
        return parts

    @property
    def text(self):
        parts = self.parts
        if len(parts) > 1 or "text" not in parts[0]:
            raise ValueError(
                "The `response.text` quick accessor only works for "
                "simple (single-`Part`) text responses. This response "
                "contains multiple `Parts`. Use the `result.parts` "
                "accessor or the full "
                "`result.candidates[index].content.parts` lookup "
                "instead"
            )
        return parts[0].text

    @property
    def prompt_feedback(self):
        return self._result.prompt_feedback


class GenerateContentResponse(BaseGenerateContentResponse):
    @classmethod
    def from_iterator(cls, iterator: Iterable[glm.GenerateContentResponse]):
        iterator = iter(iterator)
        response = next(iterator)

        return cls(
            done=False,
            iterator=iterator,
            result=response,
            chunks=[response],
        )

    @classmethod
    def from_response(cls, response: glm.GenerateContentResponse):
        return cls(
            done=True,
            iterator=None,
            result=response,
            chunks=[response],
        )

    def __iter__(self):
        # This is not thread safe.
        if self._done:
            for chunk in self._chunks:
                yield GenerateContentResponse.from_response(chunk)
            return

        # Always have the next chunk available.
        if len(self._chunks) == 0:
            self._chunks.append(next(self._iterator))

        for n in itertools.count():
            if self._error:
                raise self._error

            if n >= len(self._chunks) - 1:
                # Look ahead for a new item, so that you know the stream is done
                # when you yield the last item.
                if self._done:
                    return

                try:
                    item = next(self._iterator)
                except StopIteration:
                    self._done = True
                except Exception as e:
                    self._error = e
                    self._done = True
                else:
                    self._chunks.append(item)
                    self._result = _join_chunks([self._result, item])

            item = self._chunks[n]

            item = GenerateContentResponse.from_response(item)
            yield item

    def resolve(self):
        if self._done:
            return

        for _ in self:
            pass


class AsyncGenerateContentResponse(BaseGenerateContentResponse):
    @classmethod
    async def from_aiterator(cls, iterator: AsyncIterable[glm.GenerateContentResponse]):
        iterator = aiter(iterator)
        response = await anext(iterator)

        return cls(
            done=False,
            iterator=iterator,
            result=response,
            chunks=[response],
        )

    @classmethod
    def from_response(cls, response: glm.GenerateContentResponse):
        return cls(
            done=True,
            iterator=None,
            result=response,
            chunks=[response],
        )

    async def __aiter__(self):
        # This is not thread safe.
        if self._done:
            for chunk in self._chunks:
                yield GenerateContentResponse.from_response(chunk)
            return

        # Always have the next chunk available.
        if len(self._chunks) == 0:
            self._chunks.append(await anext(self._iterator))

        for n in itertools.count():
            if self._error:
                raise self._error

            if n >= len(self._chunks) - 1:
                # Look ahead for a new item, so that you know the stream is done
                # when you yield the last item.
                if self._done:
                    return

                try:
                    item = await anext(self._iterator)
                except StopAsyncIteration:
                    self._done = True
                except Exception as e:
                    self._error = e
                    self._done = True
                else:
                    self._chunks.append(item)
                    self._result = _join_chunks([self._result, item])

            item = self._chunks[n]

            item = GenerateContentResponse.from_response(item)
            yield item

    def resolve(self):
        if self._done:
            return

        for _ in self:
            pass
