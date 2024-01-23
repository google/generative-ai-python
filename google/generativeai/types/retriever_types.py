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

import re
import string
import abc
import dataclasses
from typing import Any, Optional, Union, Iterable, Mapping

import google.ai.generativelanguage as glm

from google.protobuf import field_mask_pb2
from google.generativeai.client import get_default_retriever_client
from google.generativeai.client import get_default_retriever_async_client
from google.generativeai import string_utils
from google.generativeai.types import safety_types
from google.generativeai.types import citation_types
from google.generativeai.types.model_types import idecode_time
from google.generativeai.utils import flatten_update_paths


_DOCUMENT_NAME_REGEX = re.compile(r"^corpora/([^/]+?)(/documents/([^/]+?))")
_CHUNK_NAME_REGEX = re.compile(r"^corpora/([^/]+?)(/documents/([^/]+?)(/chunks/([^/]+?))?)?$")
_REMOVE = string.punctuation
_REMOVE = _REMOVE.replace("-", "")  # Don't remove hyphens
_PATTERN = r"[{}]".format(_REMOVE)  # Create the pattern

Operator = glm.Condition.Operator
State = glm.Chunk.State

OperatorOptions = Union[str, int, Operator]
StateOptions = Union[str, int, State]

CreateChunkOptions = Union[Mapping[str, str], tuple[str, str]]

BatchCreateChunkOptions = Union[
    glm.BatchCreateChunksRequest,
    list[glm.CreateChunkRequest],
    Iterable[str],
    Iterable[CreateChunkOptions],
]

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
    "0": State.STATE_UNSPECIFIED,
    "state_unspecifed": State.STATE_UNSPECIFIED,
    "unspecified": State.STATE_UNSPECIFIED,
    State.STATE_PENDING_PROCESSING: State.STATE_PENDING_PROCESSING,
    "1": State.STATE_PENDING_PROCESSING,
    "pending_processing": State.STATE_PENDING_PROCESSING,
    "pending": State.STATE_PENDING_PROCESSING,
    State.STATE_ACTIVE: State.STATE_ACTIVE,
    "2": State.STATE_ACTIVE,
    "state_active": State.STATE_ACTIVE,
    "active": State.STATE_ACTIVE,
    State.STATE_FAILED: State.STATE_FAILED,
    "10": State.STATE_FAILED,  # TODO: This is specified as 10 in the proto, should it be 3 or 10?
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
class MetadataFilters:
    key: str
    conditions: Condition


@string_utils.prettyprint
@dataclasses.dataclass
class Condition:
    value: str | float


@string_utils.prettyprint
@dataclasses.dataclass
class CustomMetadata:
    key: str
    string_value: str
    string_list_value: list[str]
    numeric_value: float


@string_utils.prettyprint
@dataclasses.dataclass
class ChunkData:
    string_value: str


@string_utils.prettyprint
@dataclasses.dataclass(init=False)
class Corpus:
    """
    A `Corpus` is a collection of `Documents`.
    """

    name: str
    display_name: str

    def create_document(
        self,
        name: Optional[str] = None,
        display_name: Optional[str] = None,
        custom_metadata: Optional[list[CustomMetadata]] = None,
        client: glm.RetrieverServiceClient | None = None,
    ) -> Document:
        """
        Request to create a `Document`.

        Args:
            name: The `Document` resource name. The ID (name excluding the "corpora/*/documents/" prefix) can contain up to 40 characters
                that are lowercase alphanumeric or dashes (-). The ID cannot start or end with a dash.
            display_name: The human-readable display name for the `Document`.
            custom_metadata: User provided custom metadata stored as key-value pairs used for querying.

        Return:
            Document object with specified name or display name.

        Raises:
            ValueError: When the name is not specified or formatted incorrectly.
        """
        if client is None:
            client = get_default_retriever_client()

        if not name and not display_name:
            raise ValueError("Either the document name or display name must be specified.")

        document = None
        if name:
            if re.match(_DOCUMENT_NAME_REGEX, name):
                document = glm.Document(
                    name=name, display_name=display_name, custom_metadata=custom_metadata
                )
            elif f"corpora/{self.name}/documents/" not in name:
                document_name = f"{self.name}/documents/" + re.sub(_PATTERN, "", name)
                document = glm.Document(
                    name=document_name, display_name=display_name, custom_metadata=custom_metadata
                )
            else:
                raise ValueError(
                    f"Document name must be formatted as {self.name}/document/<document_name>."
                )

        request = glm.CreateDocumentRequest(parent=self.name, document=document)
        response = client.create_document(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        response = Document(**response)
        return response

    @string_utils.set_doc(create_document.__doc__)
    async def create_document_async(
        self,
        name: Optional[str] = None,
        display_name: Optional[str] = None,
        custom_metadata: Optional[list[CustomMetadata]] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ) -> Document:
        if client is None:
            client = get_default_retriever_async_client()

        if not name and not display_name:
            raise ValueError("Either the document name or display name must be specified.")

        document = None
        if name:
            if re.match(_DOCUMENT_NAME_REGEX, name):
                document = glm.Document(
                    name=name, display_name=display_name, custom_metadata=custom_metadata
                )
            elif f"corpora/{self.name}/documents/" not in name:
                document_name = f"{self.name}/documents/" + re.sub(_PATTERN, "", name)
                document = glm.Document(
                    name=document_name, display_name=display_name, custom_metadata=custom_metadata
                )
            else:
                raise ValueError(
                    f"Document name must be formatted as {self.name}/document/<document_name>."
                )

        request = glm.CreateDocumentRequest(parent=self.name, document=document)
        response = await client.create_document(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        response = Document(**response)
        return response

    def get_document(
        self,
        name: str,
        client: glm.RetrieverServiceClient | None = None,
    ) -> Document:
        """
        Get information about a specific `Document`.

        Args:
            name: The `Document` name.

        Return:
            `Document` of interest.
        """
        if client is None:
            client = get_default_retriever_client()

        request = glm.GetDocumentRequest(name=name)
        response = client.get_document(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        response = Document(**response)
        return response

    @string_utils.set_doc(get_document.__doc__)
    async def get_document_async(
        self,
        name: str,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ) -> Document:
        if client is None:
            client = get_default_retriever_async_client()

        request = glm.GetDocumentRequest(name=name)
        response = await client.get_document(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        response = Document(**response)
        return response

    def _apply_update(self, path, value):
        parts = path.split(".")
        for part in parts[:-1]:
            self = getattr(self, part)
        setattr(self, parts[-1], value)

    def update(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Update a list of fields for a specified `Corpus`.

        Args:
            updates: List of fields to update in a `Corpus`.

        Return:
            Updated version of the `Corpus` object.
        """
        if client is None:
            client = get_default_retriever_client()

        updates = flatten_update_paths(updates)
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        request = glm.UpdateCorpusRequest(corpus=self.to_dict(), update_mask=field_mask)
        response = client.update_corpus(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        return self

    @string_utils.set_doc(update.__doc__)
    async def update(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceAsyncClient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        updates = flatten_update_paths(updates)
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        request = glm.UpdateCorpusRequest(corpus=self.to_dict(), update_mask=field_mask)
        response = await client.update_corpus(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        return self

    def delete_document(
        self,
        name: str,
        force: Optional[bool] = None,
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Delete a document in the corpus.

        Args:
            name: The `Document` name.
            force: If set to true, any `Chunk`s and objects related to this `Document` will also be deleted.
        """
        if client is None:
            client = get_default_retriever_client()

        request = glm.DeleteDocumentRequest(name=name, force=force)
        client.delete_document(request)

    @string_utils.set_doc(delete_document.__doc__)
    async def delete_document_async(
        self,
        name: str,
        force: Optional[bool] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        request = glm.DeleteDocumentRequest(name=name, force=force)
        await client.delete_document(request)

    def list_documents(
        self,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        client: glm.RetrieverServiceClient | None = None,
    ) -> list[Document]:
        """
        List documents in corpus.

        Args:
            name: The name of the `Corpus` containing `Document`s.
            page_size: The maximum number of `Document`s to return (per page). The service may return fewer `Document`s.
            page_token: A page token, received from a previous `ListDocuments` call.

        Return:
            Paginated list of `Document`s.
        """
        if client is None:
            client = get_default_retriever_client()

        request = glm.ListDocumentsRequest(
            parent=self.name, page_size=page_size, page_token=page_token
        )
        response = client.list_documents(request)
        return response

    @string_utils.set_doc(list_documents.__doc__)
    async def list_documents_async(
        self,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ) -> list[Document]:
        if client is None:
            client = get_default_retriever_async_client()

        request = glm.ListDocumentsRequest(
            parent=self.name, page_size=page_size, page_token=page_token
        )
        response = await client.list_documents(request)
        return response

    def query_document(
        self,
        query: str,
        metadata_filters: Optional[list[str]] = None,
        results_count: Optional[int] = None,
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Query a corpus for information.

        Args:
            query: Query string to perform semantic search.
            metadata_filters: Filter for `Chunk` metadata.
            results_count: The maximum number of `Chunk`s to return.

        Returns:
            List of relevant chunks.
        """
        if client is None:
            client = get_default_retriever_client()

        if results_count:
            if results_count < 0 or results_count >= 100:
                raise ValueError("Number of results returned must be between 1 and 100.")

        request = glm.QueryCorpusRequest(
            name=self.name,
            query=query,
            metadata_filters=metadata_filters,
            results_count=results_count,
        )
        response = client.query_corpus(request)
        response = type(response).to_dict(response)

        return response

    @string_utils.set_doc(query_document.__doc__)
    async def query_document_async(
        self,
        query: str,
        metadata_filters: Optional[list[str]] = None,
        results_count: Optional[int] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        if results_count:
            if results_count < 0 or results_count >= 100:
                raise ValueError("Number of results returned must be between 1 and 100.")

        request = glm.QueryCorpusRequest(
            name=self.name,
            query=query,
            metadata_filters=metadata_filters,
            results_count=results_count,
        )
        response = await client.query_corpus(request)
        response = type(response).to_dict(response)

        return response

    def to_dict(self) -> dict[str, Any]:
        result = {"name": self.name, "display_name": self.display_name}
        return result


@string_utils.prettyprint
@dataclasses.dataclass(init=False)
class Document(abc.ABC):
    """
    A `Document` is a collection of `Chunk`s.
    """

    def __init__(self, name: str, display_name: str, custom_metadata: list[CustomMetadata]):
        self.name = name
        self.display_name = display_name
        self.custom_metadata = custom_metadata

    def create_chunk(
        self,
        name: str,
        data: ChunkData,
        custom_metadata: Optional[list[CustomMetadata]] = None,
        client: glm.RetrieverServiceClient | None = None,
    ) -> Chunk:
        """
        Create a `Chunk` object which has textual data.

        Args:
            name: The `Chunk` resource name. The ID (name excluding the "corpora/*/documents/*/chunks/" prefix) can contain up to 40 characters that are lowercase alphanumeric or dashes (-).
            data: The content for the `Chunk`, such as the text string.
            custom_metadata: User provided custom metadata stored as key-value pairs.
            state: States for the lifecycle of a `Chunk`.

        Return:
            `Chunk` object with specified data.

        Raises:
            ValueError when chunk name not specified correctly.
        """
        if client is None:
            client = get_default_retriever_client()

        if len(name) == 0:
            raise ValueError("Chunk name must be specified.")

        chunk = None
        if re.match(_CHUNK_NAME_REGEX, name):
            chunk = glm.Chunk(
                name=name, data={"string_value": data}, custom_metadata=custom_metadata
            )
        elif "chunks/" not in name:
            chunk_name = f"{self.name}/chunks/" + re.sub(_PATTERN, "", name)
            chunk = glm.Chunk(
                name=chunk_name,
                data={"string_value": data},
                custom_metadata=custom_metadata,
            )
        else:
            raise ValueError(f"Chunk name must be formatted as {self.name}/chunks/<chunk_name>.")

        request = glm.CreateChunkRequest(parent=self.name, chunk=chunk)
        response = client.create_chunk(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        response = Chunk(**response)
        return response

    @string_utils.set_doc(create_chunk.__doc__)
    async def create_chunk_async(
        self,
        name: str,
        data: ChunkData,
        custom_metadata: Optional[list[CustomMetadata]] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ) -> Chunk:
        if client is None:
            client = get_default_retriever_async_client()

        if len(name) == 0:
            raise ValueError("Chunk name must be specified.")

        chunk = None
        if re.match(_CHUNK_NAME_REGEX, name):
            chunk = glm.Chunk(
                name=name, data={"string_value": data}, custom_metadata=custom_metadata
            )
        elif "chunks/" not in name:
            chunk_name = f"{self.name}/chunks/" + re.sub(_PATTERN, "", name)
            chunk = glm.Chunk(
                name=chunk_name, data={"string_value": data}, custom_metadata=custom_metadata
            )
        else:
            raise ValueError(f"Chunk name must be formatted as {self.name}/chunks/<chunk_name>.")

        request = glm.CreateChunkRequest(parent=self.name, chunk=chunk)
        response = await client.create_chunk(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        response = Chunk(**response)
        return response

    def batch_create_chunks(
        self,
        chunks: BatchCreateChunkOptions,
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Create chunks within the given document.

        Args:
            chunks: `Chunks` to create.

        Return:
            Information about the created chunks.
        """
        if client is None:
            client = get_default_retriever_client()

        if isinstance(chunks, glm.BatchCreateChunksRequest):
            response = client.batch_create_chunks(chunks)
            response = type(response).to_dict(response)
            return response

        _requests = []
        name, data, custom_metadata = None, None, None
        if isinstance(chunks, Iterable):
            for chunk in chunks:
                if isinstance(chunk, glm.CreateChunkRequest):
                    _requests.append(chunk)
                elif isinstance(chunk, str):
                    c = glm.CreateChunkRequest(
                        parent=self.name, chunk=glm.Chunk(data={"string_value": chunk})
                    )
                    _requests.append(c)
                elif isinstance(chunk, Mapping):
                    for key, value in chunk.items():
                        if re.match(_CHUNK_NAME_REGEX, value):
                            name = value
                        elif not re.match(_CHUNK_NAME_REGEX, value) and isinstance(value, str):
                            data = chunk[key]
                        elif isinstance(value, Iterable):
                            custom_metadata = value
                        c = glm.CreateChunkRequest(  # Create a glm.CreateChunkRequest
                            parent=self.name,
                            chunk=glm.Chunk(
                                name=name,
                                data={"string_value": data},
                                custom_metadata=custom_metadata,
                            ),
                        )
                        _requests.append(c)
                elif isinstance(chunk, tuple):
                    for item in chunk:
                        if re.match(_CHUNK_NAME_REGEX, item):
                            name = item
                        elif not re.match(_CHUNK_NAME_REGEX, item) and isinstance(item, str):
                            data = item
                        elif isinstance(item, Iterable):
                            custom_metadata = item
                        c = glm.CreateChunkRequest(  # Create a glm.CreateChunkRequest
                            parent=self.name,
                            chunk=glm.Chunk(
                                name=name,
                                data={"string_value": data},
                                custom_metadata=custom_metadata,
                            ),
                        )

                else:
                    raise TypeError(
                        "Batched chunk requests must be in the format of a dictionary or tuple,"
                        "with the name as the key and the data as the value."
                    )

        request = glm.BatchCreateChunksRequest(parent=self.name, requests=_requests)
        response = client.batch_create_chunks(request)
        response = type(response).to_dict(response)
        return response

    @string_utils.set_doc(batch_create_chunks.__doc__)
    async def batch_create_chunks_async(
        self,
        chunks: BatchCreateChunkOptions,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        if isinstance(chunks, glm.BatchCreateChunksRequest):
            response = await client.batch_update_chunks(chunks)
            response = type(response).to_dict(response)
            return response

        _requests = []
        name, data, custom_metadata = None, None, None
        if isinstance(chunks, Iterable):
            for chunk in chunks:
                if isinstance(chunk, glm.CreateChunkRequest):
                    _requests.append(chunk)
                elif isinstance(chunk, str):
                    c = glm.CreateChunkRequest(
                        parent=self.name, chunk=glm.Chunk(data={"string_value": chunk})
                    )
                    _requests.append(c)
                elif isinstance(chunk, Mapping):
                    for key, value in chunk.items():
                        if re.match(_CHUNK_NAME_REGEX, value):
                            name = value
                        elif not re.match(_CHUNK_NAME_REGEX, value) and isinstance(value, str):
                            data = chunk[key]
                        elif isinstance(value, Iterable):
                            custom_metadata = value
                        c = glm.CreateChunkRequest(  # Create a glm.CreateChunkRequest
                            parent=self.name,
                            chunk=glm.Chunk(
                                name=name,
                                data={"string_value": data},
                                custom_metadata=custom_metadata,
                            ),
                        )
                        _requests.append(c)
                elif isinstance(chunk, tuple):
                    for item in chunk:
                        if re.match(_CHUNK_NAME_REGEX, item):
                            name = item
                        elif not re.match(_CHUNK_NAME_REGEX, item) and isinstance(item, str):
                            data = item
                        elif isinstance(item, Iterable):
                            custom_metadata = item
                        c = glm.CreateChunkRequest(  # Create a glm.CreateChunkRequest
                            parent=self.name,
                            chunk=glm.Chunk(
                                name=name,
                                data={"string_value": data},
                                custom_metadata=custom_metadata,
                            ),
                        )

                else:
                    raise TypeError(
                        "Batched chunk requests must be in the format of a dictionary or tuple,"
                        "with the name as the key and the data as the value."
                    )

        request = glm.BatchCreateChunksRequest(parent=self.name, requests=_requests)
        response = await client.batch_create_chunks(request)
        response = type(response).to_dict(response)
        return response

    def get_chunk(
        self,
        name: str,
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Get information about a specific chunk.

        Args:
            name: Name of `Chunk`.

        Returns:
            `Chunk` that was requested.
        """
        if client is None:
            client = get_default_retriever_client()

        request = glm.GetChunkRequest(name=name)
        response = client.get_chunk(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        response = Chunk(**response)
        return response

    @string_utils.set_doc(get_chunk.__doc__)
    async def get_chunk_async(
        self,
        name: str,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        request = glm.GetChunkRequest(name=name)
        response = await client.get_chunk(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        response = Chunk(**response)
        return response

    def list_chunks(
        self,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        List chunks of a document.

        Args:
            page_size: Maximum number of `Chunk`s to request.
            page_token: A page token, received from a previous ListChunks call.

        Return:
            List of chunks in the document.
        """
        if client is None:
            client = get_default_retriever_client()

        request = glm.ListChunksRequest(
            parent=self.name, page_size=page_size, page_token=page_token
        )
        response = client.list_chunks(request)
        return response

    @string_utils.set_doc(list_chunks.__doc__)
    async def list_chunks_async(
        self,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        client: glm.RetrieverServiceClient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        request = glm.ListChunksRequest(
            parent=self.name, page_size=page_size, page_token=page_token
        )
        response = await client.list_chunks(request)
        return response

    def _apply_update(self, path, value):
        parts = path.split(".")
        for part in parts[:-1]:
            self = getattr(self, part)
        setattr(self, parts[-1], value)

    def update_document(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Update a list of fields for a specified document.

        Args:
            updates: The list of fields to update.

        Return:
            `Chunk` object with specified updates.
        """
        if client is None:
            client = get_default_retriever_client()

        updates = flatten_update_paths(updates)
        field_mask = field_mask_pb2.FieldMask()
        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        request = glm.UpdateDocumentRequest(document=self.to_dict(), update_mask=field_mask)
        response = client.update_document(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        return self

    @string_utils.set_doc(update_document.__doc__)
    async def update_document_async(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceAsynclient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        updates = flatten_update_paths(updates)
        field_mask = field_mask_pb2.FieldMask()
        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        request = glm.UpdateDocumentRequest(document=self.to_dict(), update_mask=field_mask)
        response = await client.update_document(request)
        response = type(response).to_dict(response)
        idecode_time(response, "create_time")
        idecode_time(response, "update_time")
        return self

    def batch_update_chunks(
        self,
        chunks: BatchUpdateChunksOptions,
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Update multiple chunks within the same document.

        Args:
            chunks: Data structure specifying which `Chunk`s to update and what the required updats are.

        Return:
            Updated `Chunk`s.
        """
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
            response = client.batch_update_chunks(request)
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
            response = client.batch_update_chunks(request)
            response = type(response).to_dict(response)
            return response

    @string_utils.set_doc(batch_update_chunks.__doc__)
    async def batch_update_chunks_async(
        self,
        chunks: BatchUpdateChunksOptions,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ):
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
            response = await client.batch_update_chunks(request)
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
            response = await client.batch_update_chunks(request)
            response = type(response).to_dict(response)
            return response

    def delete_chunk(
        self,
        name: str,
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Delete a `Chunk`.

        Args:
            name: The `Chunk` name.
        """
        if client is None:
            client = get_default_retriever_client()

        request = glm.DeleteChunkRequest(name=name)
        client.delete_chunk(request)

    @string_utils.set_doc(delete_chunk.__doc__)
    async def delete_chunk_async(
        self, name: str, client: glm.RetrieverServiceAsyncClient | None = None
    ):
        if client is None:
            client = get_default_retriever_async_client()

        request = glm.DeleteChunkRequest(name=name)
        await client.delete_chunk(request)

    def batch_delete_chunks(
        self,
        chunks: BatchDeleteChunkOptions,
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Delete multiple `Chunk`s from a document.

        Args:
            chunks: Names of `Chunks` to delete.
        """
        if client is None:
            client = get_default_retriever_client()

        if all(isinstance(x, glm.DeleteChunkRequest) for x in chunks):
            request = glm.BatchDeleteChunksRequest(parent=self.name, requests=chunks)
            client.batch_delete_chunks(request)
        elif isinstance(chunks, Iterable):
            _request_list = []
            for chunk_name in chunks:
                _request_list.append(glm.DeleteChunkRequest(name=chunk_name))
            request = glm.BatchDeleteChunksRequest(parent=self.name, requests=_request_list)
            client.batch_delete_chunks(request)
        else:
            raise ValueError(
                "To delete chunks, you must pass in either the names of the chunks as an iterable, or multiple `glm.DeleteChunkRequest`s."
            )

    @string_utils.set_doc(batch_delete_chunks.__doc__)
    async def batch_delete_chunks_async(
        self,
        chunks: BatchDeleteChunkOptions,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        if all(isinstance(x, glm.DeleteChunkRequest) for x in chunks):
            request = glm.BatchDeleteChunksRequest(parent=self.name, requests=chunks)
            await client.batch_delete_chunks(request)
        elif isinstance(chunks, Iterable):
            _request_list = []
            for chunk_name in chunks:
                _request_list.append(glm.DeleteChunkRequest(name=chunk_name))
            request = glm.BatchDeleteChunksRequest(parent=self.name, requests=_request_list)
            await client.batch_delete_chunks(request)
        else:
            raise ValueError(
                "To delete chunks, you must pass in either the names of the chunks as an iterable, or multiple `glm.DeleteChunkRequest`s."
            )

    def query_document(
        self,
        query: str,
        results_count: Optional[int] = None,
        metadata_filters: Optional[list[MetadataFilters]] = None,
        client: glm.RetrieverServiceClient | None = None,
    ) -> dict:
        """
        Performs semantic search over a `Document`.

        Args:
            name: The name of the `Document` to query.
            query: Query string to perform semantic search.
            results_count: The maximum number of `Chunk`s to return. At most, 10 will be returned.
            metadata_filters: List of filter criteria for filtering for `Chunk` and `Document` metadata.

        Returns:
            Result with relevant documents from the query.
        """
        if client is None:
            client = get_default_retriever_client()

        request = glm.QueryDocumentRequest(
            name=self.name,
            query=query,
            results_count=results_count,
            metadata_filters=metadata_filters,
        )

        response = client.query_document(request)
        response = type(response).to_dict(response)

        return response

    @string_utils.set_doc(query_document.__doc__)
    async def query_document_async(
        self,
        query: str,
        results_count: Optional[int] = None,
        metadata_filters: Optional[list[MetadataFilters]] = None,
        client: glm.RetrieverServiceAsyncClient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        request = glm.QueryDocumentRequest(
            name=self.name,
            query=query,
            results_count=results_count,
            metadata_filters=metadata_filters,
        )

        response = await client.query_document(request)
        response = type(response).to_dict(response)

        return response

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "display_name": self.display_name,
            "custom_metadata": self.custom_metadata,
        }
        return result


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

    def __init__(
        self,
        name: str,
        data: ChunkData,
        custom_metadata: list[CustomMetadata] | None,
        state: State,
    ):
        self.name = name
        self.data = ChunkData(*data)
        if custom_metadata is None:
            self.custom_metadata = []
        else:
            self.custom_metadata = [CustomMetadata(*cm) for cm in custom_metadata]
        self.state = state

    def _apply_update(self, path, value):
        parts = path.split(".")
        for part in parts[:-1]:
            self = getattr(self, part)
        setattr(self, parts[-1], value)

    def update_chunk(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceClient | None = None,
    ):
        """
        Update a list of fields for a specified `Chunk`.

        Args:
            updates: List of fields to update for a `Chunk`.

        Return:
            Updated `Chunk` object.
        """
        if client is None:
            client = get_default_retriever_client()

        updates = flatten_update_paths(updates)
        field_mask = field_mask_pb2.FieldMask()
        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)
        request = glm.UpdateChunkRequest(chunk=self.to_dict(), update_mask=field_mask)
        response = client.update_chunk(request)
        response = type(response).to_dict(response)

        idecode_time(response, "create_time")
        idecode_time(response, "update_time")

        return self

    @string_utils.set_doc(update_chunk.__doc__)
    async def update_chunk_async(
        self,
        updates: dict[str, Any],
        client: glm.RetrieverServiceAsyncClient | None = None,
    ):
        if client is None:
            client = get_default_retriever_async_client()

        updates = flatten_update_paths(updates)
        field_mask = field_mask_pb2.FieldMask()
        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)
        request = glm.UpdateChunkRequest(chunk=self.to_dict(), update_mask=field_mask)
        response = await client.update_chunk(request)
        response = type(response).to_dict(response)

        idecode_time(response, "create_time")
        idecode_time(response, "update_time")

        return self

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "data": dataclasses.asdict(self.data),
            "custom_metadata": [dataclasses.asdict(cm) for cm in self.custom_metadata],
            "state": self.state,
        }
        return result