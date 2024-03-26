# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import datetime
import re
import string
import abc
import dataclasses
from typing import Any, AsyncIterable, Optional, Union, Iterable, Mapping

import google.ai.generativelanguage as glm

from google.protobuf import field_mask_pb2
from google.generativeai.client import get_default_retriever_client
from google.generativeai.client import get_default_retriever_async_client
from google.generativeai.client import get_dafault_permission_client
from google.generativeai.client import get_dafault_permission_async_client
from google.generativeai import string_utils
from google.generativeai.types import safety_types
from google.generativeai.types import citation_types
from google.generativeai.types import permission_types
from google.generativeai.types.model_types import idecode_time
from google.generativeai.utils import flatten_update_paths

_VALID_NAME = r"[a-z0-9]([a-z0-9-]{0,38}[a-z0-9])$"
NAME_ERROR_MSG = """The `name` must consist of alphanumeric characters (or -) and be 40 or fewer characters; or be empty. The name you entered:
\tlen(name)== {length}
\tname={name}
"""


def valid_name(name):
    return re.match(_VALID_NAME, name) and len(name) < 40


Operator = glm.Condition.Operator
State = glm.Chunk.State

OperatorOptions = Union[str, int, Operator]
StateOptions = Union[str, int, State]

ChunkOptions = Union[
    glm.Chunk,
    str,
    tuple[str, str],
    tuple[str, str, Any],
    Mapping[str, Any],
]  # fmt: no

BatchCreateChunkOptions = Union[
    glm.BatchCreateChunksRequest,
    Mapping[str, str],
    Mapping[str, tuple[str, str]],
    Iterable[ChunkOptions],
]  # fmt: no

UpdateChunkOptions = Union[glm.UpdateChunkRequest, Mapping[str, Any], tuple[str, Any]]

BatchUpdateChunksOptions = Union[glm.BatchUpdateChunksRequest, Iterable[UpdateChunkOptions]]

BatchDeleteChunkOptions = Union[list[glm.DeleteChunkRequest], Iterable[str]]

_OPERATOR: dict[OperatorOptions, Operator] = {
    Operator.OPERATOR_UNSPECIFIED: Operator.OPERATOR_UNSPECIFIED,
    0: Operator.OPERATOR_UNSPECIFIED,
    "operator_unspecified": Operator.OPERATOR_UNSPECIFIED,
    "unspecified": Operator.OPERATOR_UNSPECIFIED,
    Operator.LESS: Operator.LESS,
    1: Operator.LESS,
    "operator_less": Operator.LESS,
    "less": Operator.LESS,
    "<": Operator.LESS,
    Operator.LESS_EQUAL: Operator.LESS_EQUAL,
    2: Operator.LESS_EQUAL,
    "operator_less_equal": Operator.LESS_EQUAL,
    "less_equal": Operator.LESS_EQUAL,
    "<=": Operator.LESS_EQUAL,
    Operator.EQUAL: Operator.EQUAL,
    3: Operator.EQUAL,
    "operator_equal": Operator.EQUAL,
    "equal": Operator.EQUAL,
    "==": Operator.EQUAL,
    Operator.GREATER_EQUAL: Operator.GREATER_EQUAL,
    4: Operator.GREATER_EQUAL,
    "operator_greater_equal": Operator.GREATER_EQUAL,
    "greater_equal": Operator.GREATER_EQUAL,
    Operator.NOT_EQUAL: Operator.NOT_EQUAL,
    5: Operator.NOT_EQUAL,
    "operator_not_equal": Operator.NOT_EQUAL,
    "not_equal": Operator.NOT_EQUAL,
    "!=": Operator.NOT_EQUAL,
    Operator.INCLUDES: Operator.INCLUDES,
    6: Operator.INCLUDES,
    "operator_includes": Operator.INCLUDES,
    "includes": Operator.INCLUDES,
    Operator.EXCLUDES: Operator.EXCLUDES,
    6: Operator.EXCLUDES,
    "operator_excludes": Operator.EXCLUDES,
    "excludes": Operator.EXCLUDES,
    "not in": Operator.EXCLUDES,
}

_STATE: dict[StateOptions, State] = {
    State.STATE_UNSPECIFIED: State.STATE_UNSPECIFIED,
    0: State.STATE_UNSPECIFIED,
    "state_unspecifed": State.STATE_UNSPECIFIED,
    "unspecified": State.STATE_UNSPECIFIED,
    State.STATE_PENDING_PROCESSING: State.STATE_PENDING_PROCESSING,
    1: State.STATE_PENDING_PROCESSING,
    "pending_processing": State.STATE_PENDING_PROCESSING,
    "pending": State.STATE_PENDING_PROCESSING,
    State.STATE_ACTIVE: State.STATE_ACTIVE,
    2: State.STATE_ACTIVE,
    "state_active": State.STATE_ACTIVE,
    "active": State.STATE_ACTIVE,
    State.STATE_FAILED: State.STATE_FAILED,
    10: State.STATE_FAILED,
    "state_failed": State.STATE_FAILED,
    "failed": State.STATE_FAILED,
}


def to_operator(x: OperatorOptions) -> Operator:
    if isinstance(x, str):
        x = x.lower()
    return _OPERATOR[x]


def to_state(x: StateOptions) -> State:
    if isinstance(x, str):
        x = x.lower()
    return _STATE[x]


@string_utils.prettyprint
@dataclasses.dataclass
class MetadataFilter:
    key: str
    conditions: Condition


@string_utils.prettyprint
@dataclasses.dataclass
class Condition:
    value: str | float
    operation: Operator


@string_utils.prettyprint
@dataclasses.dataclass
class CustomMetadata:
    key: str
    string_value: Optional[str] = None
    string_list_value: Optional[Iterable[str]] = None
    numeric_value: Optional[float] = None


@string_utils.prettyprint
@dataclasses.dataclass
class ChunkData:
    string_value: str


def create_metadata_filters(MetadataFilter):
    metadata_filter = {
        "key": MetadataFilter.key,
        "conditions": [
            {
                "value": MetadataFilter.conditions.value,
                "operation": to_operator(MetadataFilter.conditions.operation),
            }
        ],
    }
    return metadata_filter


@string_utils.prettyprint
@dataclasses.dataclass()
class Corpus(permission_types.PermissionAdapter):
    """
    A `Corpus` is a collection of `Documents`.
    """

    name: str
    display_name: str
    create_time: datetime.datetime
    update_time: datetime.datetime

    def create_document(
        self,
        name: Optional[str] = None,
        display_name: Optional[str] = None,
        custom_metadata: Optional[list[CustomMetadata]] = None,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Document:
        """
        Request to create a `Document`.

        Args:
            name: The `Document` resource name. The ID (name excluding the "corpora/*/documents/" prefix) can contain up to 40 characters
                that are lowercase alphanumeric or dashes (-). The ID cannot start or end with a dash.
            display_name: The human-readable display name for the `Document`.
            custom_metadata: User provided custom metadata stored as key-value pairs used for querying.
            request_options: Options for the request.

        Return:
            Document object with specified name or display name.

        Raises:
            ValueError: When the name is not specified or formatted incorrectly.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        # Handle the custom_metadata parameter
        c_data = []
        if custom_metadata:
            for cm in custom_metadata:
                if cm.string_list_value:
                    c_data.append(
                        glm.CustomMetadata(
                            key=cm.key,
                            string_list_value=glm.StringList(values=cm.string_list_value),
                        )
                    )
                elif cm.string_value:
                    c_data.append(glm.CustomMetadata(key=cm.key, string_value=cm.string_value))
                elif cm.numeric_value:
                    c_data.append(glm.CustomMetadata(key=cm.key, numeric_value=cm.numeric_value))

        document, document_name = None, None
        if name is None:
            document = glm.Document(
                name=document_name, display_name=display_name, custom_metadata=custom_metadata
            )
        elif valid_name(name):
            document_name = f"{self.name}/documents/{name}"
            document = glm.Document(
                name=document_name, display_name=display_name, custom_metadata=custom_metadata
            )
        else:
            raise ValueError(NAME_ERROR_MSG.format(length=len(name), name=name))

        request = glm.CreateDocumentRequest(parent=self.name, document=document)
        response = client.create_document(request, **request_options)
        return decode_document(response)

    async def create_document_async(
        self,
        name: Optional[str] = None,
        display_name: Optional[str] = None,
        custom_metadata: Optional[list[CustomMetadata]] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Document:
        """This is the async version of `Corpus.create_document`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        # Handle the custom_metadata parameter
        c_data = []
        if custom_metadata:
            for cm in custom_metadata:
                if cm.string_list_value:
                    c_data.append(
                        glm.CustomMetadata(
                            key=cm.key,
                            string_list_value=glm.StringList(values=cm.string_list_value),
                        )
                    )
                elif cm.string_value:
                    c_data.append(glm.CustomMetadata(key=cm.key, string_value=cm.string_value))
                elif cm.numeric_value:
                    c_data.append(glm.CustomMetadata(key=cm.key, numeric_value=cm.numeric_value))

        document, document_name = None, None
        if name is None:
            document = glm.Document(
                name=document_name, display_name=display_name, custom_metadata=custom_metadata
            )
        elif valid_name(name):
            document_name = f"{self.name}/documents/{name}"
            document = glm.Document(
                name=document_name, display_name=display_name, custom_metadata=custom_metadata
            )
        else:
            raise ValueError(NAME_ERROR_MSG.format(length=len(name), name=name))

        request = glm.CreateDocumentRequest(parent=self.name, document=document)
        response = await client.create_document(request, **request_options)
        return decode_document(response)

    def get_document(
        self,
        name: str,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Document:
        """
        Get information about a specific `Document`.

        Args:
            name: The `Document` name.
            request_options: Options for the request.

        Return:
            `Document` of interest.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        if "/" not in name:
            name = f"{self.name}/documents/{name}"

        request = glm.GetDocumentRequest(name=name)
        response = client.get_document(request, **request_options)
        return decode_document(response)

    async def get_document_async(
        self,
        name: str,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Document:
        """This is the async version of `Corpus.get_document`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        if "/" not in name:
            name = f"{self.name}/documents/{name}"

        request = glm.GetDocumentRequest(name=name)
        response = await client.get_document(request, **request_options)
        return decode_document(response)

    def _apply_update(self, path, value):
        parts = path.split(".")
        for part in parts[:-1]:
            self = getattr(self, part)
        setattr(self, parts[-1], value)

    def update(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """
        Update a list of fields for a specified `Corpus`.

        Args:
            updates: List of fields to update in a `Corpus`.
            request_options: Options for the request.

        Return:
            Updated version of the `Corpus` object.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        updates = flatten_update_paths(updates)
        # At this time, only `display_name` can be updated
        for item in updates:
            if item != "display_name":
                raise ValueError("At this time, only `display_name` can be updated for `Corpus`.")
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        request = glm.UpdateCorpusRequest(corpus=self.to_dict(), update_mask=field_mask)
        client.update_corpus(request, **request_options)
        return self

    async def update_async(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """This is the async version of `Corpus.update`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        updates = flatten_update_paths(updates)
        # At this time, only `display_name` can be updated
        for item in updates:
            if item != "display_name":
                raise ValueError("At this time, only `display_name` can be updated for `Corpus`.")
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        request = glm.UpdateCorpusRequest(corpus=self.to_dict(), update_mask=field_mask)
        await client.update_corpus(request, **request_options)
        return self

    def query(
        self,
        query: str,
        metadata_filters: Optional[Iterable[MetadataFilter]] = None,
        results_count: Optional[int] = None,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Iterable[RelevantChunk]:
        """
        Query a corpus for information.

        Args:
            query: Query string to perform semantic search.
            metadata_filters: Filter for `Chunk` metadata.
            results_count: The maximum number of `Chunk`s to return; must be less than 100.
            request_options: Options for the request.

        Returns:
            List of relevant chunks.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        if results_count:
            if results_count > 100:
                raise ValueError("Number of results returned must be between 1 and 100.")

        m_f_ = []
        if metadata_filters:
            for mf in metadata_filters:
                m_f_.append(create_metadata_filters(mf))

        request = glm.QueryCorpusRequest(
            name=self.name,
            query=query,
            metadata_filters=m_f_,
            results_count=results_count,
        )
        response = client.query_corpus(request, **request_options)
        response = type(response).to_dict(response)

        # Create a RelevantChunk object for each chunk listed in response['relevant_chunks']
        relevant_chunks = []
        for c in response["relevant_chunks"]:
            rc = RelevantChunk(
                chunk_relevance_score=c["chunk_relevance_score"], chunk=Chunk(**c["chunk"])
            )
            relevant_chunks.append(rc)

        return relevant_chunks

    async def query_async(
        self,
        query: str,
        metadata_filters: Optional[Iterable[MetadataFilter]] = None,
        results_count: Optional[int] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Iterable[RelevantChunk]:
        """This is the async version of `Corpus.query`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        if results_count:
            if results_count > 100:
                raise ValueError("Number of results returned must be between 1 and 100.")

        m_f_ = []
        if metadata_filters:
            for mf in metadata_filters:
                m_f_.append(create_metadata_filters(mf))

        request = glm.QueryCorpusRequest(
            name=self.name,
            query=query,
            metadata_filters=m_f_,
            results_count=results_count,
        )
        response = await client.query_corpus(request, **request_options)
        response = type(response).to_dict(response)

        # Create a RelevantChunk object for each chunk listed in response['relevant_chunks']
        relevant_chunks = []
        for c in response["relevant_chunks"]:
            rc = RelevantChunk(
                chunk_relevance_score=c["chunk_relevance_score"], chunk=Chunk(**c["chunk"])
            )
            relevant_chunks.append(rc)

        return relevant_chunks

    def delete_document(
        self,
        name: str,
        force: bool = False,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """
        Delete a document in the corpus.

        Args:
            name: The `Document` name.
            force: If set to true, any `Chunk`s and objects related to this `Document` will also be deleted.
            request_options: Options for the request.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        if "/" not in name:
            name = f"{self.name}/documents/{name}"

        request = glm.DeleteDocumentRequest(name=name, force=bool(force))
        client.delete_document(request, **request_options)

    async def delete_document_async(
        self,
        name: str,
        force: bool = False,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """This is the async version of `Corpus.delete_document`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        if "/" not in name:
            name = f"{self.name}/documents/{name}"

        request = glm.DeleteDocumentRequest(name=name, force=bool(force))
        await client.delete_document(request, **request_options)

    def list_documents(
        self,
        page_size: Optional[int] = None,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Iterable[Document]:
        """
        List documents in corpus.

        Args:
            name: The name of the `Corpus` containing `Document`s.
            page_size: The maximum number of `Document`s to return (per page). The service may return fewer `Document`s.
            request_options: Options for the request.

        Return:
            Paginated list of `Document`s.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        request = glm.ListDocumentsRequest(
            parent=self.name,
            page_size=page_size,
        )
        for doc in client.list_documents(request, **request_options):
            yield decode_document(doc)

    async def list_documents_async(
        self,
        page_size: Optional[int] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> AsyncIterable[Document]:
        """This is the async version of `Corpus.list_documents`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        request = glm.ListDocumentsRequest(
            parent=self.name,
            page_size=page_size,
        )
        async for doc in await client.list_documents(request, **request_options):
            yield decode_document(doc)

    def transfer_ownership(
        self,
        email_address: str,
        client: glm.PermissionServiceClient | None = None,
    ) -> None:
        raise NotImplementedError("This method is not implemented for class Corpus")

    async def transfer_ownership_async(
        self,
        email_address: str,
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> None:
        raise NotImplementedError("This method is not implemented for class Corpus")

    def to_dict(self) -> dict[str, Any]:
        result = {"name": self.name, "display_name": self.display_name}
        return result


def decode_document(document):
    document = type(document).to_dict(document)
    idecode_time(document, "create_time")
    idecode_time(document, "update_time")
    return Document(**document)


@string_utils.prettyprint
@dataclasses.dataclass()
class Document(abc.ABC):
    """
    A `Document` is a collection of `Chunk`s.
    """

    name: str
    display_name: str
    custom_metadata: list[CustomMetadata]
    create_time: datetime.datetime
    update_time: datetime.datetime

    def create_chunk(
        self,
        data: str | ChunkData,
        name: Optional[str] = None,
        custom_metadata: Optional[list[CustomMetadata]] = None,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Chunk:
        """
        Create a `Chunk` object which has textual data.

        Args:
            data: The content for the `Chunk`, such as the text string.
            name: The `Chunk` resource name. The ID (name excluding the "corpora/*/documents/*/chunks/" prefix) can contain up to 40 characters that are lowercase alphanumeric or dashes (-).
            custom_metadata: User provided custom metadata stored as key-value pairs.
            state: States for the lifecycle of a `Chunk`.
            request_options: Options for the request.

        Return:
            `Chunk` object with specified data.

        Raises:
            ValueError when chunk name not specified correctly.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        chunk_name, chunk = None, None
        if name is None:
            chunk_name = None
        elif valid_name(name):
            chunk_name = f"{self.name}/chunks/{name}"
        else:
            raise ValueError(NAME_ERROR_MSG.format(length=len(name), name=name))

        # Handle the custom_metadata parameter
        c_data = []
        if custom_metadata:
            for cm in custom_metadata:
                if cm.string_list_value:
                    c_data.append(
                        glm.CustomMetadata(
                            key=cm.key,
                            string_list_value=glm.StringList(values=cm.string_list_value),
                        )
                    )
                elif cm.string_value:
                    c_data.append(glm.CustomMetadata(key=cm.key, string_value=cm.string_value))
                elif cm.numeric_value:
                    c_data.append(glm.CustomMetadata(key=cm.key, numeric_value=cm.numeric_value))

        if isinstance(data, str):
            chunk = glm.Chunk(name=chunk_name, data={"string_value": data}, custom_metadata=c_data)
        else:
            chunk = glm.Chunk(
                name=chunk_name,
                data={"string_value": data},
                custom_metadata=c_data,
            )

        request = glm.CreateChunkRequest(parent=self.name, chunk=chunk)
        response = client.create_chunk(request, **request_options)
        return decode_chunk(response)

    async def create_chunk_async(
        self,
        data: str | ChunkData,
        name: Optional[str] = None,
        custom_metadata: Optional[list[CustomMetadata]] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Chunk:
        """This is the async version of `Document.create_chunk`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        chunk_name, chunk = None, None
        if name is None:
            chunk_name = None
        elif valid_name(name):
            chunk_name = f"{self.name}/chunks/{name}"
        else:
            raise ValueError(NAME_ERROR_MSG.format(length=len(name), name=name))

        # Handle the custom_metadata parameter
        c_data = []
        if custom_metadata:
            for cm in custom_metadata:
                if cm.string_list_value:
                    c_data.append(
                        glm.CustomMetadata(
                            key=cm.key,
                            string_list_value=glm.StringList(values=cm.string_list_value),
                        )
                    )
                elif cm.string_value:
                    c_data.append(glm.CustomMetadata(key=cm.key, string_value=cm.string_value))
                elif cm.numeric_value:
                    c_data.append(glm.CustomMetadata(key=cm.key, numeric_value=cm.numeric_value))

        if isinstance(data, str):
            chunk = glm.Chunk(name=chunk_name, data={"string_value": data}, custom_metadata=c_data)
        else:
            chunk = glm.Chunk(
                name=chunk_name,
                data={"string_value": data},
                custom_metadata=c_data,
            )

        request = glm.CreateChunkRequest(parent=self.name, chunk=chunk)
        response = await client.create_chunk(request, **request_options)
        return decode_chunk(response)

    def _make_chunk(self, chunk: ChunkOptions) -> glm.Chunk:
        # del self
        if isinstance(chunk, glm.Chunk):
            return glm.Chunk(chunk)
        elif isinstance(chunk, str):
            return glm.Chunk(data={"string_value": chunk})
        elif isinstance(chunk, tuple):
            if len(chunk) == 2:
                name, data = chunk  # pytype: disable=bad-unpacking
                custom_metadata = None
            elif len(chunk) == 3:
                name, data, custom_metadata = chunk  # pytype: disable=bad-unpacking
            else:
                raise ValueError(
                    f"Tuples should have length 2 or 3, got length: {len(chunk)}\n"
                    f"value: {chunk}"
                )

            return glm.Chunk(
                name=name,
                data={"string_value": data},
                custom_metadata=custom_metadata,
            )
        elif isinstance(chunk, Mapping):
            if isinstance(chunk["data"], str):
                chunk = dict(chunk)
                chunk["data"] = {"string_value": chunk["data"]}
            return glm.Chunk(chunk)
        else:
            raise TypeError(
                f"Could not convert instance of `{type(chunk)}` chunk:" f"value: {chunk}"
            )

    def _make_batch_create_chunk_request(
        self, chunks: BatchCreateChunkOptions
    ) -> glm.BatchCreateChunksRequest:
        if isinstance(chunks, glm.BatchCreateChunksRequest):
            return chunks

        if isinstance(chunks, Mapping):
            chunks = chunks.items()
            chunks = (
                # Flatten tuples
                (key,) + value if isinstance(value, tuple) else (key, value)
                for key, value in chunks
            )

        requests = []
        for i, chunk in enumerate(chunks):
            chunk = self._make_chunk(chunk)
            if chunk.name == "":
                chunk.name = str(i)

            chunk.name = f"{self.name}/chunks/{chunk.name}"

            requests.append(glm.CreateChunkRequest(parent=self.name, chunk=chunk))

        return glm.BatchCreateChunksRequest(parent=self.name, requests=requests)

    def batch_create_chunks(
        self,
        chunks: BatchCreateChunkOptions,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """
        Create chunks within the given document.

        Args:
            chunks: `Chunks` to create.
            request_options: Options for the request.

        Return:
            Information about the created chunks.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        request = self._make_batch_create_chunk_request(chunks)
        response = client.batch_create_chunks(request, **request_options)
        return [decode_chunk(chunk) for chunk in response.chunks]

    async def batch_create_chunks_async(
        self,
        chunks: BatchCreateChunkOptions,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """This is the async version of `Document.batch_create_chunk`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        request = self._make_batch_create_chunk_request(chunks)
        response = await client.batch_create_chunks(request, **request_options)
        return [decode_chunk(chunk) for chunk in response.chunks]

    def get_chunk(
        self,
        name: str,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """
        Get information about a specific chunk.

        Args:
            name: Name of `Chunk`.
            request_options: Options for the request.

        Returns:
            `Chunk` that was requested.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        if "/" not in name:
            name = f"{self.name}/chunks/{name}"

        request = glm.GetChunkRequest(name=name)
        response = client.get_chunk(request, **request_options)
        return decode_chunk(response)

    async def get_chunk_async(
        self,
        name: str,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """This is the async version of `Document.get_chunk`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        if "/" not in name:
            name = f"{self.name}/chunks/{name}"

        request = glm.GetChunkRequest(name=name)
        response = await client.get_chunk(request, **request_options)
        return decode_chunk(response)

    def list_chunks(
        self,
        page_size: Optional[int] = None,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> Iterable[Chunk]:
        """
        List chunks of a document.

        Args:
            page_size: Maximum number of `Chunk`s to request.
            request_options: Options for the request.

        Return:
            List of chunks in the document.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        request = glm.ListChunksRequest(parent=self.name, page_size=page_size)
        for chunk in client.list_chunks(request, **request_options):
            yield decode_chunk(chunk)

    async def list_chunks_async(
        self,
        page_size: Optional[int] = None,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> AsyncIterable[Chunk]:
        """This is the async version of `Document.list_chunks`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        request = glm.ListChunksRequest(parent=self.name, page_size=page_size)
        async for chunk in await client.list_chunks(request, **request_options):
            yield decode_chunk(chunk)

    def query(
        self,
        query: str,
        metadata_filters: Optional[Iterable[MetadataFilter]] = None,
        results_count: Optional[int] = None,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> list[RelevantChunk]:
        """
        Query a `Document` in the `Corpus` for information.

        Args:
            query: Query string to perform semantic search.
            metadata_filters: Filter for `Chunk` metadata.
            results_count: The maximum number of `Chunk`s to return.

        Returns:
            List of relevant chunks.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        if results_count:
            if results_count < 0 or results_count >= 100:
                raise ValueError("Number of results returned must be between 1 and 100.")

        m_f_ = []
        if metadata_filters:
            for mf in metadata_filters:
                m_f_.append(create_metadata_filters(mf))

        request = glm.QueryDocumentRequest(
            name=self.name,
            query=query,
            metadata_filters=m_f_,
            results_count=results_count,
        )
        response = client.query_document(request, **request_options)
        response = type(response).to_dict(response)

        # Create a RelevantChunk object for each chunk listed in response['relevant_chunks']
        relevant_chunks = []
        for c in response["relevant_chunks"]:
            rc = RelevantChunk(
                chunk_relevance_score=c["chunk_relevance_score"], chunk=Chunk(**c["chunk"])
            )
            relevant_chunks.append(rc)

        return relevant_chunks

    async def query_async(
        self,
        query: str,
        metadata_filters: Optional[Iterable[MetadataFilter]] = None,
        results_count: Optional[int] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> list[RelevantChunk]:
        """This is the async version of `Document.query`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        if results_count:
            if results_count < 0 or results_count >= 100:
                raise ValueError("Number of results returned must be between 1 and 100.")

        m_f_ = []
        if metadata_filters:
            for mf in metadata_filters:
                m_f_.append(create_metadata_filters(mf))

        request = glm.QueryDocumentRequest(
            name=self.name,
            query=query,
            metadata_filters=m_f_,
            results_count=results_count,
        )
        response = await client.query_document(request, **request_options)
        response = type(response).to_dict(response)

        # Create a RelevantChunk object for each chunk listed in response['relevant_chunks']
        relevant_chunks = []
        for c in response["relevant_chunks"]:
            rc = RelevantChunk(
                chunk_relevance_score=c["chunk_relevance_score"], chunk=Chunk(**c["chunk"])
            )
            relevant_chunks.append(rc)

        return relevant_chunks

    def _apply_update(self, path, value):
        parts = path.split(".")
        for part in parts[:-1]:
            self = getattr(self, part)
        setattr(self, parts[-1], value)

    def update(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """
        Update a list of fields for a specified document.

        Args:
            updates: The list of fields to update.
            request_options: Options for the request.

        Return:
            `Chunk` object with specified updates.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        updates = flatten_update_paths(updates)
        # At this time, only `display_name` can be updated
        for item in updates:
            if item != "display_name":
                raise ValueError("At this time, only `display_name` can be updated for `Document`.")
        field_mask = field_mask_pb2.FieldMask()
        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        request = glm.UpdateDocumentRequest(document=self.to_dict(), update_mask=field_mask)
        client.update_document(request, **request_options)
        return self

    async def update_async(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """This is the async version of `Document.update`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        updates = flatten_update_paths(updates)
        # At this time, only `display_name` can be updated
        for item in updates:
            if item != "display_name":
                raise ValueError("At this time, only `display_name` can be updated for `Document`.")
        field_mask = field_mask_pb2.FieldMask()
        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        request = glm.UpdateDocumentRequest(document=self.to_dict(), update_mask=field_mask)
        await client.update_document(request, **request_options)
        return self

    def batch_update_chunks(
        self,
        chunks: BatchUpdateChunksOptions,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """
        Update multiple chunks within the same document.

        Args:
            chunks: Data structure specifying which `Chunk`s to update and what the required updats are.
            request_options: Options for the request.

        Return:
            Updated `Chunk`s.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        # TODO (@snkancharla): Add idecode_time here in each conditional loop?
        if isinstance(chunks, glm.BatchUpdateChunksRequest):
            response = client.batch_update_chunks(chunks)
            response = type(response).to_dict(response)
            return response

        _requests = []
        if isinstance(chunks, Mapping):
            # Key is name of chunk, value is a dictionary of updates
            for key, value in chunks.items():
                c = self.get_chunk(name=key)
                updates = flatten_update_paths(value)
                field_mask = field_mask_pb2.FieldMask()
                for path in updates.keys():
                    field_mask.paths.append(path)
                for path, value in updates.items():
                    c._apply_update(path, value)
                _requests.append(glm.UpdateChunkRequest(chunk=c.to_dict(), update_mask=field_mask))
            request = glm.BatchUpdateChunksRequest(parent=self.name, requests=_requests)
            response = client.batch_update_chunks(request, **request_options)
            response = type(response).to_dict(response)
            return response
        if isinstance(chunks, Iterable) and not isinstance(chunks, Mapping):
            for chunk in chunks:
                if isinstance(chunk, glm.UpdateChunkRequest):
                    _requests.append(chunk)
                elif isinstance(chunk, tuple):
                    # First element is name of chunk, second element contains updates
                    c = self.get_chunk(name=chunk[0])
                    updates = flatten_update_paths(chunk[1])
                    field_mask = field_mask_pb2.FieldMask()
                    for path in updates.keys():
                        field_mask.paths.append(path)
                    for path, value in updates.items():
                        c._apply_update(path, value)
                    _requests.append({"chunk": c.to_dict(), "update_mask": field_mask})
                else:
                    raise TypeError(
                        "The `chunks` parameter must be a list of glm.UpdateChunkRequests,"
                        "dictionaries, or tuples of dictionaries."
                    )
            request = glm.BatchUpdateChunksRequest(parent=self.name, requests=_requests)
            response = client.batch_update_chunks(request, **request_options)
            response = type(response).to_dict(response)
            return response

    async def batch_update_chunks_async(
        self,
        chunks: BatchUpdateChunksOptions,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """This is the async version of `Document.batch_update_chunks`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        # TODO (@snkancharla): Add idecode_time here in each conditional loop?
        if isinstance(chunks, glm.BatchUpdateChunksRequest):
            response = await client.batch_update_chunks(chunks)
            response = type(response).to_dict(response)
            return response

        _requests = []
        if isinstance(chunks, Mapping):
            # Key is name of chunk, value is a dictionary of updates
            for key, value in chunks.items():
                c = self.get_chunk(name=key)
                updates = flatten_update_paths(value)
                field_mask = field_mask_pb2.FieldMask()
                for path in updates.keys():
                    field_mask.paths.append(path)
                for path, value in updates.items():
                    c._apply_update(path, value)
                _requests.append(glm.UpdateChunkRequest(chunk=c.to_dict(), update_mask=field_mask))
            request = glm.BatchUpdateChunksRequest(parent=self.name, requests=_requests)
            response = await client.batch_update_chunks(request, **request_options)
            response = type(response).to_dict(response)
            return response
        if isinstance(chunks, Iterable) and not isinstance(chunks, Mapping):
            for chunk in chunks:
                if isinstance(chunk, glm.UpdateChunkRequest):
                    _requests.append(chunk)
                elif isinstance(chunk, tuple):
                    # First element is name of chunk, second element contains updates
                    c = self.get_chunk(name=chunk[0])
                    updates = flatten_update_paths(chunk[1])
                    field_mask = field_mask_pb2.FieldMask()
                    for path in updates.keys():
                        field_mask.paths.append(path)
                    for path, value in updates.items():
                        c._apply_update(path, value)
                    _requests.append({"chunk": c.to_dict(), "update_mask": field_mask})
                else:
                    raise TypeError(
                        "The `chunks` parameter must be a list of glm.UpdateChunkRequests,"
                        "dictionaries, or tuples of dictionaries."
                    )
            request = glm.BatchUpdateChunksRequest(parent=self.name, requests=_requests)
            response = await client.batch_update_chunks(request, **request_options)
            response = type(response).to_dict(response)
            return response

    def delete_chunk(
        self,
        name: str,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,  # fmt: {}
    ):
        """
        Delete a `Chunk`.

        Args:
            name: The `Chunk` name.
            request_options: Options for the request.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        if "/" not in name:
            name = f"{self.name}/chunks/{name}"

        request = glm.DeleteChunkRequest(name=name)
        client.delete_chunk(request, **request_options)

    async def delete_chunk_async(
        self,
        name: str,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,  # fmt: {}
    ):
        """This is the async version of `Document.delete_chunk`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        if "/" not in name:
            name = f"{self.name}/chunks/{name}"

        request = glm.DeleteChunkRequest(name=name)
        await client.delete_chunk(request, **request_options)

    def batch_delete_chunks(
        self,
        chunks: BatchDeleteChunkOptions,
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """
        Delete multiple `Chunk`s from a document.

        Args:
            chunks: Names of `Chunks` to delete.
            request_options: Options for the request.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        if all(isinstance(x, glm.DeleteChunkRequest) for x in chunks):
            request = glm.BatchDeleteChunksRequest(parent=self.name, requests=chunks)
            client.batch_delete_chunks(request, **request_options)
        elif isinstance(chunks, Iterable):
            _request_list = []
            for chunk_name in chunks:
                _request_list.append(glm.DeleteChunkRequest(name=chunk_name))
            request = glm.BatchDeleteChunksRequest(parent=self.name, requests=_request_list)
            client.batch_delete_chunks(request, **request_options)
        else:
            raise ValueError(
                "To delete chunks, you must pass in either the names of the chunks as an iterable, or multiple `glm.DeleteChunkRequest`s."
            )

    async def batch_delete_chunks_async(
        self,
        chunks: BatchDeleteChunkOptions,
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """This is the async version of `Document.batch_delete_chunks`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        if all(isinstance(x, glm.DeleteChunkRequest) for x in chunks):
            request = glm.BatchDeleteChunksRequest(parent=self.name, requests=chunks)
            await client.batch_delete_chunks(request, **request_options)
        elif isinstance(chunks, Iterable):
            _request_list = []
            for chunk_name in chunks:
                _request_list.append(glm.DeleteChunkRequest(name=chunk_name))
            request = glm.BatchDeleteChunksRequest(parent=self.name, requests=_request_list)
            await client.batch_delete_chunks(request, **request_options)
        else:
            raise ValueError(
                "To delete chunks, you must pass in either the names of the chunks as an iterable, or multiple `glm.DeleteChunkRequest`s."
            )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "display_name": self.display_name,
            "custom_metadata": self.custom_metadata,
        }
        return result


def decode_chunk(chunk: glm.Chunk) -> Chunk:
    chunk = type(chunk).to_dict(chunk)
    idecode_time(chunk, "create_time")
    idecode_time(chunk, "update_time")
    return Chunk(**chunk)


@string_utils.prettyprint
@dataclasses.dataclass
class RelevantChunk:
    chunk_relevance_score: float
    chunk: Chunk


@string_utils.prettyprint
@dataclasses.dataclass(init=False)
class Chunk(abc.ABC):
    """
    A `Chunk` is part of the `Document`, or the actual text.
    """

    name: str
    data: ChunkData
    custom_metadata: list[CustomMetadata] | None
    state: State
    create_time: datetime.datetime | None
    update_time: datetime.datetime | None

    def __init__(
        self,
        name: str,
        data: ChunkData | str,
        custom_metadata: Iterable[CustomMetadata] | None,
        state: State,
        create_time: datetime.datetime | str | None = None,
        update_time: datetime.datetime | str | None = None,
    ):
        self.name = name
        if isinstance(data, str):
            self.data = ChunkData(string_value=data)
        elif isinstance(data, dict):
            self.data = ChunkData(string_value=data["string_value"])
        if custom_metadata is None:
            self.custom_metadata = []
        else:
            self.custom_metadata = [CustomMetadata(*cm) for cm in custom_metadata]

        self.state = to_state(state)

        if create_time is None:
            self.create_time = None
        elif isinstance(create_time, datetime.datetime):
            self.create_time = create_time
        else:
            self.create_time = datetime.datetime.strptime(create_time, "%Y-%m-%dT%H:%M:%S.%fZ")

        if update_time is None:
            self.update_time = None
        elif isinstance(update_time, datetime.datetime):
            self.update_time = update_time
        else:
            self.update_time = datetime.datetime.strptime(update_time, "%Y-%m-%dT%H:%M:%S.%fZ")

    def _apply_update(self, path, value):
        parts = path.split(".")
        for part in parts[:-1]:
            self = getattr(self, part)
        setattr(self, parts[-1], value)

    def update(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """
        Update a list of fields for a specified `Chunk`.

        Args:
            updates: List of fields to update for a `Chunk`.
            request_options: Options for the request.

        Return:
            Updated `Chunk` object.
        """
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_client()

        updates = flatten_update_paths(updates)
        # At this time, only `data` can be updated
        for item in updates:
            if item != "data.string_value":
                raise ValueError(
                    f"At this time, only `data` can be updated for `Chunk`. Got {item}."
                )
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)
        request = glm.UpdateChunkRequest(chunk=self.to_dict(), update_mask=field_mask)
        client.update_chunk(request, **request_options)
        return self

    async def update_async(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceAsyncClient | None = None,
        request_options: dict[str, Any] | None = None,
    ):
        """This is the async version of `Chunk.update`."""
        if request_options is None:
            request_options = {}

        if client is None:
            client = get_default_retriever_async_client()

        updates = flatten_update_paths(updates)
        # At this time, only `data` can be updated
        for item in updates:
            if item != "data.string_value":
                raise ValueError(
                    f"At this time, only `data` can be updated for `Chunk`. Got {item}."
                )
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)
        request = glm.UpdateChunkRequest(chunk=self.to_dict(), update_mask=field_mask)
        await client.update_chunk(request, **request_options)
        return self

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "data": dataclasses.asdict(self.data),
            "custom_metadata": [dataclasses.asdict(cm) for cm in self.custom_metadata],
            "state": self.state,
        }
        return result
