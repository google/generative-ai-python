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
import unittest
import unittest.mock as mock

import google.ai.generativelanguage as glm

from google.generativeai import retriever
from google.generativeai import client
from google.generativeai.types import retriever_types as retriever_service
from absl.testing import absltest
from absl.testing import parameterized


class UnitTests(parameterized.TestCase):
    def setUp(self):
        self.client = unittest.mock.MagicMock()

        client._client_manager.clients["retriever"] = self.client

        self.observed_requests = []

        self.responses = {}

        def add_client_method(f):
            name = f.__name__
            setattr(self.client, name, f)
            return f

        @add_client_method
        def create_corpus(
            request: glm.CreateCorpusRequest,
        ) -> glm.Corpus:
            self.observed_requests.append(request)
            return glm.Corpus(
                name="corpora/demo_corpus",
                display_name="demo-corpus",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def get_corpus(
            request: glm.GetCorpusRequest,
        ) -> glm.Corpus:
            self.observed_requests.append(request)
            return glm.Corpus(
                name="corpora/demo_corpus",
                display_name="demo-corpus",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def update_corpus(
            request: glm.UpdateCorpusRequest,
        ) -> glm.Corpus:
            self.observed_requests.append(request)
            return glm.Corpus(
                name="corpora/demo-corpus",
                display_name="demo-corpus-1",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def list_corpora(
            request: glm.ListCorporaRequest,
        ) -> glm.ListCorporaResponse:
            self.observed_requests.append(request)
            return [
                glm.Corpus(
                    name="corpora/demo_corpus-1",
                    display_name="demo-corpus-1",
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                ),
                glm.Corpus(
                    name="corpora/demo-corpus-2",
                    display_name="demo-corpus-2",
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                ),
            ]

        @add_client_method
        def query_corpus(
            request: glm.QueryCorpusRequest,
        ) -> glm.QueryCorpusResponse:
            self.observed_requests.append(request)
            return glm.QueryCorpusResponse(
                relevant_chunks=[
                    glm.RelevantChunk(
                        chunk_relevance_score=0.08,
                        chunk=glm.Chunk(
                            name="corpora/demo-corpus/documents/demo-doc/chunks/demo-chunk",
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
        def delete_corpus(request: glm.DeleteCorpusRequest) -> None:
            self.observed_requests.append(request)

        @add_client_method
        def create_document(
            request: glm.CreateDocumentRequest,
        ) -> retriever_service.Document:
            self.observed_requests.append(request)
            return glm.Document(
                name="corpora/demo-corpus/documents/demo-doc",
                display_name="demo-doc",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def get_document(
            request: glm.GetDocumentRequest,
        ) -> retriever_service.Document:
            self.observed_requests.append(request)
            return glm.Document(
                name="corpora/demo-corpus/documents/demo_doc",
                display_name="demo-doc",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def update_document(
            request: glm.UpdateDocumentRequest,
        ) -> glm.Document:
            self.observed_requests.append(request)
            return glm.Document(
                name="corpora/demo-corpus/documents/demo_doc",
                display_name="demo-doc-1",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def list_documents(
            request: glm.ListDocumentsRequest,
        ) -> glm.ListDocumentsResponse:
            self.observed_requests.append(request)
            return [
                glm.Document(
                    name="corpora/demo-corpus/documents/demo_doc_1",
                    display_name="demo-doc-1",
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                ),
                glm.Document(
                    name="corpora/demo-corpus/documents/demo_doc_2",
                    display_name="demo-doc-2",
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                ),
            ]

        @add_client_method
        def delete_document(
            request: glm.DeleteDocumentRequest,
        ) -> None:
            self.observed_requests.append(request)

        @add_client_method
        def query_document(
            request: glm.QueryDocumentRequest,
        ) -> glm.QueryDocumentResponse:
            self.observed_requests.append(request)
            return glm.QueryDocumentResponse(
                relevant_chunks=[
                    glm.RelevantChunk(
                        chunk_relevance_score=0.08,
                        chunk=glm.Chunk(
                            name="demo-chunk",
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
        def create_chunk(
            request: glm.CreateChunkRequest,
        ) -> retriever_service.Chunk:
            self.observed_requests.append(request)
            return glm.Chunk(
                name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                data={"string_value": "This is a demo chunk."},
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def batch_create_chunks(
            request: glm.BatchCreateChunksRequest,
        ) -> glm.BatchCreateChunksResponse:
            self.observed_requests.append(request)
            return glm.BatchCreateChunksResponse(
                chunks=[
                    glm.Chunk(
                        name="corpora/demo-corpus/documents/demo-doc/chunks/dc",
                        data={"string_value": "This is a demo chunk."},
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                    glm.Chunk(
                        name="corpora/demo-corpus/documents/demo-doc/chunks/dc1",
                        data={"string_value": "This is another demo chunk."},
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                ]
            )

        @add_client_method
        def get_chunk(
            request: glm.GetChunkRequest,
        ) -> retriever_service.Chunk:
            self.observed_requests.append(request)
            return glm.Chunk(
                name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                data={"string_value": "This is a demo chunk."},
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def list_chunks(
            request: glm.ListChunksRequest,
        ) -> glm.ListChunksResponse:
            self.observed_requests.append(request)
            return [
                glm.Chunk(
                    name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                    data={"string_value": "This is a demo chunk."},
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                ),
                glm.Chunk(
                    name="corpora/demo-corpus/documents/demo-doc/chunks/demo-chunk-1",
                    data={"string_value": "This is another demo chunk."},
                    create_time="2000-01-01T01:01:01.123456Z",
                    update_time="2000-01-01T01:01:01.123456Z",
                ),
            ]

        @add_client_method
        def update_chunk(request: glm.UpdateChunkRequest) -> glm.Chunk:
            self.observed_requests.append(request)
            return glm.Chunk(
                name="corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk",
                data={"string_value": "This is an updated demo chunk."},
                custom_metadata=[
                    glm.CustomMetadata(
                        key="tags",
                        string_list_value=glm.StringList(
                            values=["Google For Developers", "Project IDX", "Blog", "Announcement"]
                        ),
                    )
                ],
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def batch_update_chunks(
            request: glm.BatchUpdateChunksRequest,
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
                        name="corpora/demo-corpus/documents/demo-doc/chunks/demo-chunk-1",
                        data={"string_value": "This is another updated chunk."},
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                ]
            )

        @add_client_method
        def delete_chunk(
            request: glm.DeleteChunkRequest,
        ) -> None:
            self.observed_requests.append(request)

        @add_client_method
        def batch_delete_chunks(
            request: glm.BatchDeleteChunksRequest,
        ) -> None:
            self.observed_requests.append(request)

    def test_create_corpus(self, name="demo-corpus"):
        x = retriever.create_corpus(name=name)
        self.assertIsInstance(x, retriever_service.Corpus)
        self.assertEqual("demo-corpus", x.display_name)
        self.assertEqual("corpora/demo_corpus", x.name)

    def test_create_corpus_no_name(self):
        x = retriever.create_corpus()
        self.assertIsInstance(x, retriever_service.Corpus)
        self.assertEqual("demo-corpus", x.display_name)
        self.assertEqual("corpora/demo_corpus", x.name)

    def test_get_corpus(self, name="demo-corpus"):
        x = retriever.create_corpus(name=name)
        c = retriever.get_corpus(name=x.name)
        self.assertEqual("demo-corpus", c.display_name)

    def test_update_corpus(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        update_request = demo_corpus.update(updates={"display_name": "demo-corpus_1"})
        self.assertIsInstance(self.observed_requests[-1], glm.UpdateCorpusRequest)
        self.assertEqual("demo-corpus_1", demo_corpus.display_name)

    def test_list_corpora(self):
        x = list(retriever.list_corpora(page_size=1))
        self.assertEqual(len(x), 2)

    def test_query_corpus(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        demo_chunk = demo_document.create_chunk(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        q = demo_corpus.query(query="What kind of chunk is this?")
        self.assertEqual(
            q,
            [
                retriever_service.RelevantChunk(
                    chunk_relevance_score=0.08,
                    chunk=retriever_service.Chunk(
                        name="corpora/demo-corpus/documents/demo-doc/chunks/demo-chunk",
                        data="This is a demo chunk.",
                        custom_metadata=[],
                        state=0,
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                )
            ],
        )

    def test_delete_corpus(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        delete_request = retriever.delete_corpus(name="corpora/demo_corpus", force=True)
        self.assertIsInstance(self.observed_requests[-1], glm.DeleteCorpusRequest)

    def test_create_document(self, display_name="demo-doc"):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        x = demo_corpus.create_document(name=display_name)
        self.assertIsInstance(x, retriever_service.Document)
        self.assertEqual("demo-doc", x.display_name)

    def test_create_document_no_name(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        x = demo_corpus.create_document()
        self.assertIsInstance(x, retriever_service.Document)
        self.assertEqual("demo-doc", x.display_name)

    def test_get_document(self, display_name="demo-doc"):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        x = demo_corpus.create_document(name=display_name)
        d = demo_corpus.get_document(name=x.name)
        self.assertEqual("demo-doc", d.display_name)

    def test_update_document(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        update_request = demo_document.update(updates={"display_name": "demo-doc-1"})
        self.assertEqual("demo-doc-1", demo_document.display_name)

    def test_delete_document(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        demo_doc2 = demo_corpus.create_document(name="demo-doc-2")
        delete_request = demo_corpus.delete_document(name="corpora/demo-corpus/documents/demo_doc")
        self.assertIsInstance(self.observed_requests[-1], glm.DeleteDocumentRequest)

    def test_list_documents(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        demo_doc2 = demo_corpus.create_document(name="demo-doc-2")
        self.assertLen(list(demo_corpus.list_documents()), 2)

    def test_query_document(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        demo_chunk = demo_document.create_chunk(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        q = demo_document.query(query="What kind of chunk is this?")
        self.assertEqual(
            q,
            [
                retriever_service.RelevantChunk(
                    chunk_relevance_score=0.08,
                    chunk=retriever_service.Chunk(
                        name="demo-chunk",
                        data="This is a demo chunk.",
                        custom_metadata=[],
                        state=0,
                        create_time="2000-01-01T01:01:01.123456Z",
                        update_time="2000-01-01T01:01:01.123456Z",
                    ),
                )
            ],
        )

    def test_create_chunk(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        x = demo_document.create_chunk(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        self.assertIsInstance(x, retriever_service.Chunk)
        self.assertEqual("corpora/demo-corpus/documents/dem-doc/chunks/demo-chunk", x.name)
        self.assertEqual(retriever_service.ChunkData("This is a demo chunk."), x.data)

    def test_create_chunk_empty(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        x = demo_document.create_chunk(
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
                        "name": "corpora/demo-corpus/documents/demo-doc/chunks/dc",
                        "data": "This is a demo chunk.",
                    },
                    {
                        "name": "corpora/demo-corpus/documents/demo-doc/chunks/dc1",
                        "data": "This is another demo chunk.",
                    },
                ],
            ),
            dict(
                testcase_name="tuples",
                chunks=[
                    (
                        "corpora/demo-corpus/documents/demo-doc/chunks/dc",
                        "This is a demo chunk.",
                    ),
                    (
                        "corpora/demo-corpus/documents/demo-doc/chunks/dc1",
                        "This is another demo chunk.",
                    ),
                ],
            ),
        ]
    )
    def test_batch_create_chunks(self, chunks):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        chunks = demo_document.batch_create_chunks(chunks=chunks)
        self.assertIsInstance(self.observed_requests[-1], glm.BatchCreateChunksRequest)
        self.assertEqual("This is a demo chunk.", chunks[0].data.string_value)
        self.assertEqual("This is another demo chunk.", chunks[1].data.string_value)

    def test_get_chunk(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        x = demo_document.create_chunk(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        ch = demo_document.get_chunk(name=x.name)
        self.assertEqual(retriever_service.ChunkData("This is a demo chunk."), ch.data)

    def test_list_chunks(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        x = demo_document.create_chunk(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        y = demo_document.create_chunk(
            name="demo-chunk-1",
            data="This is another demo chunk.",
        )

        list_req = list(demo_document.list_chunks())
        self.assertIsInstance(self.observed_requests[-1], glm.ListChunksRequest)
        self.assertLen(list_req, 2)

    def test_update_chunk(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        x = demo_document.create_chunk(
            name="demo-chunk",
            data="This is a demo chunk.",
            custom_metadata=[
                retriever_service.CustomMetadata(
                    key="tag",
                    value=[
                        "Google For Developers",
                        "Project IDX",
                        "Blog",
                        "Announcement",
                    ],
                )
            ],
        )
        x.update(updates={"data": {"string_value": "This is an updated demo chunk."}})
        self.assertEqual(
            retriever_service.ChunkData("This is an updated demo chunk."),
            x.data,
        )

    @parameterized.named_parameters(
        [
            dict(
                testcase_name="dictionary_of_updates",
                updates={
                    "corpora/demo-corpus/documents/demo-doc/chunks/demo-chunk": {
                        "data": {"string_value": "This is an updated chunk."}
                    },
                    "corpora/demo-corpus/documents/demo-doc/chunks/demo-chunk-1": {
                        "data": {"string_value": "This is another updated chunk."}
                    },
                },
            ),
            dict(
                testcase_name="list_of_tuples",
                updates=[
                    (
                        "corpora/demo-corpus/documents/demo-doc/chunks/demo-chunk",
                        {"data": {"string_value": "This is an updated chunk."}},
                    ),
                    (
                        "corpora/demo-corpus/documents/demo-doc/chunks/demo-chunk-1",
                        {"data": {"string_value": "This is another updated chunk."}},
                    ),
                ],
            ),
        ],
    )
    def test_batch_update_chunks_data_structures(self, updates):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        x = demo_document.create_chunk(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        y = demo_document.create_chunk(
            name="demo-chunk-1",
            data="This is another demo chunk.",
        )
        update_request = demo_document.batch_update_chunks(chunks=updates)
        self.assertIsInstance(self.observed_requests[-1], glm.BatchUpdateChunksRequest)
        self.assertEqual(
            "This is an updated chunk.", update_request["chunks"][0]["data"]["string_value"]
        )
        self.assertEqual(
            "This is another updated chunk.", update_request["chunks"][1]["data"]["string_value"]
        )

    def test_delete_chunk(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        x = demo_document.create_chunk(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        delete_request = demo_document.delete_chunk(name="demo-chunk")
        self.assertIsInstance(self.observed_requests[-1], glm.DeleteChunkRequest)

    def test_batch_delete_chunks(self):
        demo_corpus = retriever.create_corpus(name="demo-corpus")
        demo_document = demo_corpus.create_document(name="demo-doc")
        x = demo_document.create_chunk(
            name="demo-chunk",
            data="This is a demo chunk.",
        )
        y = demo_document.create_chunk(
            name="demo-chunk",
            data="This is another demo chunk.",
        )
        delete_request = demo_document.batch_delete_chunks(chunks=[x.name, y.name])
        self.assertIsInstance(self.observed_requests[-1], glm.BatchDeleteChunksRequest)

    @parameterized.named_parameters(
        [
            "create_corpus",
            retriever.create_corpus,
            retriever.create_corpus_async,
        ],
        [
            "get_corpus",
            retriever.get_corpus,
            retriever.get_corpus_async,
        ],
        [
            "delete_corpus",
            retriever.delete_corpus,
            retriever.delete_corpus_async,
        ],
        [
            "list_corpora",
            retriever.list_corpora,
            retriever.list_corpora_async,
        ],
        [
            "Corpus.create_document",
            retriever_service.Corpus.create_document,
            retriever_service.Corpus.create_document_async,
        ],
        [
            "Corpus.get_document",
            retriever_service.Corpus.get_document,
            retriever_service.Corpus.get_document_async,
        ],
        [
            "Corpus.update",
            retriever_service.Corpus.update,
            retriever_service.Corpus.update_async,
        ],
        [
            "Corpus.query",
            retriever_service.Corpus.query,
            retriever_service.Corpus.query_async,
        ],
        [
            "Corpus.list_documents",
            retriever_service.Corpus.list_documents,
            retriever_service.Corpus.list_documents_async,
        ],
        [
            "Corpus.delete_document",
            retriever_service.Corpus.delete_document,
            retriever_service.Corpus.delete_document_async,
        ],
        [
            "Document.create_chunk",
            retriever_service.Document.create_chunk,
            retriever_service.Document.create_chunk_async,
        ],
        [
            "Document.get_chunk",
            retriever_service.Document.get_chunk,
            retriever_service.Document.get_chunk_async,
        ],
        [
            "Document.batch_create_chunks",
            retriever_service.Document.batch_create_chunks,
            retriever_service.Document.batch_create_chunks_async,
        ],
        [
            "Document.list_chunks",
            retriever_service.Document.list_chunks,
            retriever_service.Document.list_chunks_async,
        ],
        [
            "Document.query",
            retriever_service.Document.query,
            retriever_service.Document.query_async,
        ],
        [
            "Document.update",
            retriever_service.Document.update,
            retriever_service.Document.update_async,
        ],
        [
            "Document.batch_update_chunks",
            retriever_service.Document.batch_update_chunks,
            retriever_service.Document.batch_update_chunks_async,
        ],
        [
            "Document.delete_chunk",
            retriever_service.Document.delete_chunk,
            retriever_service.Document.delete_chunk_async,
        ],
        [
            "Document.batch_delete_chunks",
            retriever_service.Document.batch_delete_chunks,
            retriever_service.Document.batch_delete_chunks_async,
        ],
        [
            "Chunk.update",
            retriever_service.Chunk.update,
            retriever_service.Chunk.update_async,
        ],
    )
    def test_async_code_match(self, obj, aobj):
        import inspect
        import re

        source = inspect.getsource(obj)
        asource = inspect.getsource(aobj)
        source = re.sub('""".*"""', "", source, flags=re.DOTALL)
        asource = re.sub('""".*"""', "", asource, flags=re.DOTALL)
        asource = (
            asource.replace("anext", "next")
            .replace("aiter", "iter")
            .replace("_async", "")
            .replace("async ", "")
            .replace("await ", "")
            .replace("Async", "")
            .replace("ASYNC_", "")
        )

        asource = re.sub(" *?# type: ignore", "", asource)
        self.assertEqual(source, asource)


if __name__ == "__main__":
    absltest.main()
