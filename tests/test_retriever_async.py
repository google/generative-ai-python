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
import collections
import copy
import math
from typing import Any
import unittest
import unittest.mock as mock

import google.ai.generativelanguage as glm

from google.generativeai import retriever
from google.generativeai import client as client_lib
from google.generativeai.types import retriever_types as retriever_service
from absl.testing import absltest
from absl.testing import parameterized


class AsyncTests(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = unittest.mock.AsyncMock()

        client_lib._client_manager.clients["retriever_async"] = self.client

        def add_client_method(f):
            name = f.__name__
            setattr(self.client, name, f)
            return f

        self.observed_requests = []
        self.responses = collections.defaultdict(list)

        @add_client_method
        async def create_corpus(
            request: glm.CreateCorpusRequest,
            **kwargs,
        ) -> glm.Corpus:
            self.observed_requests.append(request)
            return glm.Corpus(
                name="corpora/demo-corpus",
                display_name="demo-corpus",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        async def get_corpus(
            request: glm.GetCorpusRequest,
            **kwargs,
        ) -> glm.Corpus:
            self.observed_requests.append(request)
            return glm.Corpus(
                name="corpora/demo-corpus",
                display_name="demo-corpus",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        async def update_corpus(
            request: glm.UpdateCorpusRequest,
            **kwargs,
        ) -> glm.Corpus:
            self.observed_requests.append(request)
            return glm.Corpus(
                name="corpora/demo-corpus",
                display_name="demo-corpus-1",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        async def list_corpora(
            request: glm.ListCorporaRequest,
            **kwargs,
        ) -> glm.ListCorporaResponse:
            self.observed_requests.append(request)

            async def results():
                yield glm.Corpus(
                    name="corpora/demo-corpus-1",
                    display_name="demo-corpus-1",
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                )
                yield glm.Corpus(
                    name="corpora/demo-corpus_2",
                    display_name="demo-corpus-2",
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                )

            return results()

        @add_client_method
        async def query_corpus(
            request: glm.QueryCorpusRequest,
            **kwargs,
        ) -> glm.QueryCorpusResponse:
            self.observed_requests.append(request)
            return glm.QueryCorpusResponse(
                relevant_chunks=[
                    glm.RelevantChunk(
                        chunk_relevance_score=0.08,
                        chunk=glm.Chunk(
                            name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                            data={"string_value": "This is a demo chunk."},
                            custom_metadata=[],
                            state=0,
                            create_time="2000-01-01T01:01:01.123456Z",
                            update_time="2000-01-01T01:01:01.123456Z",
                        ),
                    )
                ]
            )

        @add_client_method
        async def delete_corpus(
            request: glm.DeleteCorpusRequest,
            **kwargs,
        ) -> None:
            self.observed_requests.append(request)

        @add_client_method
        async def create_document(
            request: glm.CreateDocumentRequest,
            **kwargs,
        ) -> retriever_service.Document:
            self.observed_requests.append(request)
            return glm.Document(
                name="corpora/demo-corpus/documents/demo-doc",
                display_name="demo-doc",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        async def get_document(
            request: glm.GetDocumentRequest,
            **kwargs,
        ) -> retriever_service.Document:
            self.observed_requests.append(request)
            return glm.Document(
                name="corpora/demo-corpus/documents/demo-doc",
                display_name="demo-doc",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        async def update_document(
            request: glm.UpdateDocumentRequest,
            **kwargs,
        ) -> glm.Document:
            self.observed_requests.append(request)
            return glm.Document(
                name="corpora/demo-corpus/documents/demo-doc",
                display_name="demo-doc-1",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        async def list_documents(
            request: glm.ListDocumentsRequest,
            **kwargs,
        ) -> glm.ListDocumentsResponse:
            self.observed_requests.append(request)

            async def results():
                yield glm.Document(
                    name="corpora/demo-corpus/documents/dem-doc_1",
                    display_name="demo-doc-1",
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                )
                yield glm.Document(
                    name="corpora/demo-corpus/documents/dem-doc_2",
                    display_name="demo-doc_2",
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                )

            return results()

        @add_client_method
        async def delete_document(
            request: glm.DeleteDocumentRequest,
            **kwargs,
        ) -> None:
            self.observed_requests.append(request)

        @add_client_method
        async def query_document(
            request: glm.QueryDocumentRequest,
            **kwargs,
        ) -> glm.QueryDocumentResponse:
            self.observed_requests.append(request)
            return glm.QueryDocumentResponse(
                relevant_chunks=[
                    glm.RelevantChunk(
                        chunk_relevance_score=0.08,
                        chunk=glm.Chunk(
                            name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                            data={"string_value": "This is a demo chunk."},
                            custom_metadata=[],
                            state=0,
                            create_time="2000-01-01T01:01:01.123456Z",
                            update_time="2000-01-01T01:01:01.123456Z",
                        ),
                    )
                ]
            )

        @add_client_method
        async def create_chunk(
            request: glm.CreateChunkRequest,
            **kwargs,
        ) -> retriever_service.Chunk:
            self.observed_requests.append(request)
            return glm.Chunk(
                name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                data={"string_value": "This is a demo chunk."},
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        async def batch_create_chunks(
            request: glm.BatchCreateChunksRequest,
            **kwargs,
        ) -> glm.BatchCreateChunksResponse:
            self.observed_requests.append(request)
            return glm.BatchCreateChunksResponse(
                chunks=[
                    glm.Chunk(
                        name="corpora/demo-corpus/documents/dem-doc/chunks/dc",
                        data={"string_value": "This is a demo chunk."},
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                    glm.Chunk(
                        name="corpora/demo-corpus/documents/dem-doc/chunks/dc1",
                        data={"string_value": "This is another demo chunk."},
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                ]
            )

        @add_client_method
        async def get_chunk(
            request: glm.GetChunkRequest,
            **kwargs,
        ) -> retriever_service.Chunk:
            self.observed_requests.append(request)
            return glm.Chunk(
                name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                data={"string_value": "This is a demo chunk."},
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        async def list_chunks(
            request: glm.ListChunksRequest,
            **kwargs,
        ) -> glm.ListChunksResponse:
            self.observed_requests.append(request)

            async def results():
                yield glm.Chunk(
                    name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                    data={"string_value": "This is a demo chunk."},
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                )
                yield glm.Chunk(
                    name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk-1",
                    data={"string_value": "This is another demo chunk."},
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                )

            return results()

        @add_client_method
        async def update_chunk(
            request: glm.UpdateChunkRequest,
            **kwargs,
        ) -> glm.Chunk:
            self.observed_requests.append(request)
            return glm.Chunk(
                name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                data={"string_value": "This is an updated demo chunk."},
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        async def batch_update_chunks(
            request: glm.BatchUpdateChunksRequest,
            **kwargs,
        ) -> glm.BatchUpdateChunksResponse:
            self.observed_requests.append(request)
            return glm.BatchUpdateChunksResponse(
                chunks=[
                    glm.Chunk(
                        name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                        data={"string_value": "This is an updated chunk."},
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                    glm.Chunk(
                        name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk-1",
                        data={"string_value": "This is another updated chunk."},
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                ]
            )

        @add_client_method
        async def delete_chunk(
            request: glm.DeleteChunkRequest,
            **kwargs,
        ) -> None:
            self.observed_requests.append(request)

        @add_client_method
        async def batch_delete_chunks(
            request: glm.BatchDeleteChunksRequest,
            **kwargs,
        ) -> None:
            self.observed_requests.append(request)

    async def test_create_corpus(self, name="demo-corpus"):
        x = await retriever.create_corpus_async(name=name)
        self.assertIsInstance(x, retriever_service.Corpus)
        self.assertEqual("demo-corpus", x.display_name)
        self.assertEqual("corpora/demo-corpus", x.name)

    async def test_get_corpus(self, name="demo-corpus"):
        x = await retriever.create_corpus_async(name=name)
        c = await retriever.get_corpus_async(name=x.name)
        self.assertEqual("demo-corpus", c.display_name)

    async def test_update_corpus(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        update_request = await demo_corpus.update_async(updates={"display_name": "demo_corpus_1"})
        self.assertEqual("demo_corpus_1", demo_corpus.display_name)

    async def test_list_corpora(self):
        result = []
        async for x in retriever.list_corpora_async(page_size=1):
            result.append(x)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    async def test_query_corpus(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        demo_chunk = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        q = await demo_corpus.query_async(query="What kind of chunk is this?")
        self.assertEqual(
            q,
            [
                retriever_service.RelevantChunk(
                    chunk_relevance_score=0.08,
                    chunk=retriever_service.Chunk(
                        name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                        data="This is a demo chunk.",
                        custom_metadata=[],
                        state=0,
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                )
            ],
        )

    async def test_delete_corpus(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        delete_request = await retriever.delete_corpus_async(name="corpora/demo-corpus", force=True)
        self.assertIsInstance(self.observed_requests[-1], glm.DeleteCorpusRequest)

    async def test_create_document(self, display_name="demo-doc"):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        x = await demo_corpus.create_document_async(name=display_name)
        self.assertIsInstance(x, retriever_service.Document)
        self.assertEqual("demo-doc", x.display_name)

    async def test_get_document(self, display_name="demo-doc"):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        x = await demo_corpus.create_document_async(name=display_name)
        d = await demo_corpus.get_document_async(name=x.name)
        self.assertEqual("demo-doc", d.display_name)

    async def test_update_document(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        update_request = await demo_document.update_async(updates={"display_name": "demo-doc-1"})
        self.assertEqual("demo-doc-1", demo_document.display_name)

    async def test_delete_document(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        demo_doc2 = await demo_corpus.create_document_async(name="demo-doc-2")
        delete_request = await demo_corpus.delete_document_async(
            name="corpora/demo-corpus/documents/demo-doc"
        )
        self.assertIsInstance(self.observed_requests[-1], glm.DeleteDocumentRequest)

    async def test_list_documents(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        demo_doc2 = await demo_corpus.create_document_async(name="demo-doc-2")
        self.assertLen(list(demo_corpus.list_documents()), 2)

    async def test_query_document(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        demo_chunk = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        q = await demo_document.query_async(query="What kind of chunk is this?")
        self.assertEqual(
            q,
            [
                retriever_service.RelevantChunk(
                    chunk_relevance_score=0.08,
                    chunk=retriever_service.Chunk(
                        name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                        data="This is a demo chunk.",
                        custom_metadata=[],
                        state=0,
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                )
            ],
        )

    async def test_create_chunk(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        x = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        self.assertIsInstance(x, retriever_service.Chunk)
        self.assertEqual("corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk", x.name)
        self.assertEqual(retriever_service.ChunkData("This is a demo chunk."), x.data)

    async def test_create_chunk_empty(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        x = await demo_document.create_chunk_async(
            data="This is a demo chunk.",
        )
        self.assertIsInstance(x, retriever_service.Chunk)
        self.assertEqual("corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk", x.name)
        self.assertEqual(retriever_service.ChunkData("This is a demo chunk."), x.data)

    @parameterized.named_parameters(
        [
            dict(
                testcase_name="dictionaries",
                chunks=[
                    {
                        "name": "corpora/demo-corpus/documents/dem-doc/chunks/dc",
                        "data": "This is a demo chunk.",
                    },
                    {
                        "name": "corpora/demo-corpus/documents/dem-doc/chunks/dc1",
                        "data": "This is another demo chunk.",
                    },
                ],
            ),
            dict(
                testcase_name="tuples",
                chunks=[
                    (
                        "corpora/demo-corpus/documents/dem-doc/chunks/dc",
                        "This is a demo chunk.",
                    ),
                    (
                        "corpora/demo-corpus/documents/dem-doc/chunks/dc1",
                        "This is another demo chunk.",
                    ),
                ],
            ),
        ]
    )
    async def test_batch_create_chunks(self, chunks):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        chunks = await demo_document.batch_create_chunks_async(chunks=chunks)
        self.assertIsInstance(self.observed_requests[-1], glm.BatchCreateChunksRequest)
        self.assertEqual("This is a demo chunk.", chunks[0].data.string_value)
        self.assertEqual("This is another demo chunk.", chunks[1].data.string_value)

    async def test_get_chunk(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        x = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        ch = await demo_document.get_chunk_async(name=x.name)
        self.assertEqual(retriever_service.ChunkData("This is a demo chunk."), ch.data)

    async def test_list_chunks(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        x = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        y = await demo_document.create_chunk_async(
            name="demo-chunk-1",
            data="This is another demo chunk.",
        )
        chunks = []
        async for chunk in demo_document.list_chunks_async():
            chunks.append(chunk)
        self.assertIsInstance(self.observed_requests[-1], glm.ListChunksRequest)
        self.assertLen(chunks, 2)

    async def test_update_chunk(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        x = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        await x.update_async(updates={"data": {"string_value": "This is an updated demo chunk."}})
        self.assertEqual(
            retriever_service.ChunkData("This is an updated demo chunk."),
            x.data,
        )

    @parameterized.named_parameters(
        [
            dict(
                testcase_name="dictionary_of_updates",
                updates={
                    "corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk": {
                        "data": {"string_value": "This is an updated chunk."}
                    },
                    "corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk-1": {
                        "data": {"string_value": "This is another updated chunk."}
                    },
                },
            ),
            dict(
                testcase_name="list_of_tuples",
                updates=[
                    (
                        "corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                        {"data": {"string_value": "This is an updated chunk."}},
                    ),
                    (
                        "corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk-1",
                        {"data": {"string_value": "This is another updated chunk."}},
                    ),
                ],
            ),
        ],
    )
    async def test_batch_update_chunks_data_structures(self, updates):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        x = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        y = await demo_document.create_chunk_async(
            name="demo-chunk-1",
            data="This is another demo chunk.",
        )
        update_request = await demo_document.batch_update_chunks_async(chunks=updates)
        self.assertIsInstance(self.observed_requests[-1], glm.BatchUpdateChunksRequest)
        self.assertEqual(
            "This is an updated chunk.", update_request["chunks"][0]["data"]["string_value"]
        )
        self.assertEqual(
            "This is another updated chunk.", update_request["chunks"][1]["data"]["string_value"]
        )

    async def test_delete_chunk(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        x = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        delete_request = await demo_document.delete_chunk_async(
            name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk"
        )
        self.assertIsInstance(self.observed_requests[-1], glm.DeleteChunkRequest)

    async def test_batch_delete_chunks(self):
        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        demo_document = await demo_corpus.create_document_async(name="demo-doc")
        x = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        y = await demo_document.create_chunk_async(
            name="demo-chunk",
            data="This is another demo chunk.",
        )
        delete_request = await demo_document.batch_delete_chunks_async(chunks=[x.name, y.name])
        self.assertIsInstance(self.observed_requests[-1], glm.BatchDeleteChunksRequest)

    async def test_get_corpus_called_with_request_options(self):
        self.client.get_corpus = unittest.mock.AsyncMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        try:
            x = await retriever.create_corpus_async(name="demo-corpus")
            await retriever.get_corpus_async(name=x.name, request_options=request_options)
        except AttributeError:
            pass

        self.client.get_corpus.assert_called_once_with(request, **request_options)

    async def test_update_corpus_called_with_request_options(self):
        self.client.update_corpus = unittest.mock.AsyncMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        demo_corpus = await retriever.create_corpus_async(name="demo-corpus")
        await demo_corpus.update_async(
            updates={"display_name": "demo_corpus_1"}, request_options=request_options
        )

        self.client.update_corpus.assert_called_once_with(request, **request_options)


if __name__ == "__main__":
    absltest.main()
