# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
import copy
import math
from typing import Any
import unittest
import unittest.mock as mock

import google.ai.generativelanguage as glm

from google.generativeai import embedding

from google.generativeai import client as client_lib
from absl.testing import absltest
from absl.testing import parameterized

DEFAULT_EMB_MODEL = "models/embedding-001"


class AsyncTests(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = unittest.mock.AsyncMock()

        client_lib._client_manager.clients["generative_async"] = self.client

        self.observed_requests = []

        def add_client_method(f):
            name = f.__name__
            setattr(self.client, name, f)
            return f

        @add_client_method
        async def embed_content(
            request: glm.EmbedContentRequest,
            **kwargs,
        ) -> glm.EmbedContentResponse:
            self.observed_requests.append(request)
            return glm.EmbedContentResponse(embedding=glm.ContentEmbedding(values=[1, 2, 3]))

        @add_client_method
        async def batch_embed_contents(
            request: glm.BatchEmbedContentsRequest,
            **kwargs,
        ) -> glm.BatchEmbedContentsResponse:
            self.observed_requests.append(request)
            return glm.BatchEmbedContentsResponse(
                embeddings=[glm.ContentEmbedding(values=[1, 2, 3])] * len(request.requests)
            )

    async def test_embed_content_async(self):
        text = "What are you?"
        emb = await embedding.embed_content_async(model=DEFAULT_EMB_MODEL, content=text)

        self.assertIsInstance(emb, dict)
        self.assertEqual(
            self.observed_requests[-1],
            glm.EmbedContentRequest(
                model=DEFAULT_EMB_MODEL, content=glm.Content(parts=[glm.Part(text="What are you?")])
            ),
        )
        self.assertIsInstance(emb["embedding"][0], float)

    @parameterized.named_parameters(
        [
            dict(
                testcase_name="even-batch",
                batch_size=100,
            ),
            dict(
                testcase_name="even-batch-plus-one",
                batch_size=101,
            ),
            dict(testcase_name="odd-batch", batch_size=237),
        ]
    )
    async def test_batch_embed_contents_async(self, batch_size):
        text = ["What are you?"]
        texts = text * batch_size
        emb = await embedding.embed_content_async(model=DEFAULT_EMB_MODEL, content=texts)

        self.assertIsInstance(emb, dict)

        # Check that the list has the right length.
        self.assertIsInstance(emb["embedding"][0], list)
        self.assertLen(emb["embedding"], len(texts))

        # Check that the right number of requests were sent.
        self.assertLen(
            self.observed_requests,
            math.ceil(len(texts) / embedding.EMBEDDING_MAX_BATCH_SIZE),
        )

    async def test_embed_content_async_title_and_task_1(self):
        text = "What are you?"
        emb = await embedding.embed_content_async(
            model=DEFAULT_EMB_MODEL,
            content=text,
            task_type="retrieval_document",
            title="Exploring AI",
        )

        self.assertEqual(
            embedding.to_task_type("retrieval_document"),
            embedding.EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        )

    async def test_embed_content_async_title_and_task_2(self):
        text = "What are you?"
        with self.assertRaises(ValueError):
            await embedding.embed_content_async(
                model=DEFAULT_EMB_MODEL, content=text, task_type="unspecified", title="Exploring AI"
            )

    async def test_embed_content_called_with_request_options(self):
        self.client.embed_content = unittest.mock.AsyncMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        try:
            text = "What are you?"
            emb = await embedding.embed_content_async(
                model=DEFAULT_EMB_MODEL, content=text, request_options=request_options
            )
        except AttributeError:
            pass

        self.client.embed_content.assert_called_once_with(request, **request_options)


if __name__ == "__main__":
    absltest.main()
