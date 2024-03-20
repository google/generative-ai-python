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
import copy
import math
from typing import Any
import unittest
import unittest.mock as mock

import google.ai.generativelanguage as glm

from google.generativeai import text as text_service
from google.generativeai import client
from google.generativeai.types import safety_types
from google.generativeai.types import model_types
from absl.testing import absltest
from absl.testing import parameterized


class UnitTests(parameterized.TestCase):
    def setUp(self):
        self.client = unittest.mock.MagicMock()

        client._client_manager.clients["text"] = self.client
        client._client_manager.clients["model"] = self.client

        self.observed_requests = []

        self.responses = {}

        def add_client_method(f):
            name = f.__name__
            setattr(self.client, name, f)
            return f

        @add_client_method
        def generate_text(
            request: glm.GenerateTextRequest,
            **kwargs,
        ) -> glm.GenerateTextResponse:
            self.observed_requests.append(request)
            return self.responses["generate_text"]

        @add_client_method
        def embed_text(
            request: glm.EmbedTextRequest,
            **kwargs,
        ) -> glm.EmbedTextResponse:
            self.observed_requests.append(request)
            return self.responses["embed_text"]

        @add_client_method
        def batch_embed_text(
            request: glm.EmbedTextRequest,
            **kwargs,
        ) -> glm.EmbedTextResponse:
            self.observed_requests.append(request)

            return glm.BatchEmbedTextResponse(
                embeddings=[glm.Embedding(value=[1, 2, 3])] * len(request.texts)
            )

        @add_client_method
        def count_text_tokens(
            request: glm.CountTextTokensRequest,
            **kwargs,
        ) -> glm.CountTextTokensResponse:
            self.observed_requests.append(request)
            return self.responses["count_text_tokens"]

        @add_client_method
        def get_tuned_model(name) -> glm.TunedModel:
            request = glm.GetTunedModelRequest(name=name)
            self.observed_requests.append(request)
            response = copy.copy(self.responses["get_tuned_model"])
            return response

    @parameterized.named_parameters(
        [
            dict(testcase_name="string", prompt="Hello how are"),
        ]
    )
    def test_make_prompt(self, prompt):
        x = text_service._make_text_prompt(prompt)
        self.assertIsInstance(x, glm.TextPrompt)
        self.assertEqual("Hello how are", x.text)

    @parameterized.named_parameters(
        [
            dict(testcase_name="string", prompt="What are you"),
        ]
    )
    def test_make_generate_text_request(self, prompt):
        x = text_service._make_generate_text_request(model="models/chat-bison-001", prompt=prompt)
        self.assertEqual("models/chat-bison-001", x.model)
        self.assertIsInstance(x, glm.GenerateTextRequest)

    @parameterized.named_parameters(
        [
            dict(
                testcase_name="basic_model",
                model="models/chat-lamda-001",
                text="What are you?",
            )
        ]
    )
    def test_generate_embeddings(self, model, text):
        self.responses["embed_text"] = glm.EmbedTextResponse(
            embedding=glm.Embedding(value=[1, 2, 3])
        )

        emb = text_service.generate_embeddings(model=model, text=text)

        self.assertIsInstance(emb, dict)
        self.assertEqual(self.observed_requests[-1], glm.EmbedTextRequest(model=model, text=text))
        self.assertIsInstance(emb["embedding"][0], float)

    @parameterized.named_parameters(
        [
            dict(
                testcase_name="small-2",
                model="models/chat-lamda-001",
                text=["Who are you?", "Who am I?"],
            ),
            dict(
                testcase_name="even-batch",
                model="models/chat-lamda-001",
                text=["Who are you?"] * 100,
            ),
            dict(
                testcase_name="even-batch-plus-one",
                model="models/chat-lamda-001",
                text=["Who are you?"] * 101,
            ),
            dict(
                testcase_name="odd-batch",
                model="models/chat-lamda-001",
                text=["Who are you?"] * 237,
            ),
        ]
    )
    def test_generate_embeddings_batch(self, model, text):
        emb = text_service.generate_embeddings(model=model, text=text)

        self.assertIsInstance(emb, dict)

        # Check first and last requests.
        self.assertEqual(self.observed_requests[-1].model, model)
        self.assertEqual(self.observed_requests[-1].texts[-1], text[-1])
        self.assertEqual(self.observed_requests[0].texts[0], text[0])

        # Check that the list has the right length.
        self.assertIsInstance(emb["embedding"][0], list)
        self.assertLen(emb["embedding"], len(text))

        # Check that the right number of requests were sent.
        self.assertLen(
            self.observed_requests,
            math.ceil(len(text) / text_service.EMBEDDING_MAX_BATCH_SIZE),
        )

    @parameterized.named_parameters(
        [
            dict(testcase_name="basic", prompt="Why did the chicken cross the"),
            dict(
                testcase_name="temperature",
                prompt="Why did the chicken cross the",
                temperature=0.75,
            ),
            dict(
                testcase_name="stop_list",
                prompt="Why did the chicken cross the",
                stop_sequences=["a", "b", "c"],
            ),
            dict(
                testcase_name="count",
                prompt="Why did the chicken cross the",
                candidate_count=2,
            ),
        ]
    )
    def test_generate_response(self, *, prompt, **kwargs):
        self.responses["generate_text"] = glm.GenerateTextResponse(
            candidates=[
                glm.TextCompletion(output=" road?"),
                glm.TextCompletion(output=" bridge?"),
                glm.TextCompletion(output=" river?"),
            ]
        )

        complete = text_service.generate_text(prompt=prompt, **kwargs)

        self.assertEqual(
            self.observed_requests[-1],
            glm.GenerateTextRequest(
                model="models/text-bison-001", prompt=glm.TextPrompt(text=prompt), **kwargs
            ),
        )

        self.assertIsInstance(complete.result, str)

        self.assertEqual(
            complete.candidates,
            [
                {"output": " road?", "safety_ratings": []},
                {"output": " bridge?", "safety_ratings": []},
                {"output": " river?", "safety_ratings": []},
            ],
        )

    def test_stop_string(self):
        self.responses["generate_text"] = glm.GenerateTextResponse(
            candidates=[
                glm.TextCompletion(output="Hello world?"),
                glm.TextCompletion(output="Hell!"),
                glm.TextCompletion(output="I'm going to stop"),
            ]
        )
        complete = text_service.generate_text(prompt="Hello", stop_sequences="stop")

        self.assertEqual(
            self.observed_requests[-1],
            glm.GenerateTextRequest(
                model="models/text-bison-001",
                prompt=glm.TextPrompt(text="Hello"),
                stop_sequences=["stop"],
            ),
        )
        # Just make sure it made it into the request object.
        self.assertEqual(self.observed_requests[-1].stop_sequences, ["stop"])

    @parameterized.named_parameters(
        [
            dict(
                testcase_name="basic",
                safety_settings=[
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                        "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                        "threshold": safety_types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    },
                ],
            ),
            dict(
                testcase_name="strings",
                safety_settings=[
                    {
                        "category": "medical",
                        "threshold": "block_none",
                    },
                    {
                        "category": "violent",
                        "threshold": "low",
                    },
                ],
            ),
            dict(
                testcase_name="flat",
                safety_settings={"medical": "block_none", "sex": "low"},
            ),
            dict(
                testcase_name="mixed",
                safety_settings={
                    "medical": safety_types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE: 1,
                },
            ),
        ]
    )
    def test_safety_settings(self, safety_settings):
        self.responses["generate_text"] = glm.GenerateTextResponse(
            candidates=[
                glm.TextCompletion(output="No"),
            ]
        )
        # This test really just checks that the safety_settings get converted to a proto.
        result = text_service.generate_text(
            prompt="Say something wicked.", safety_settings=safety_settings
        )

        self.assertEqual(
            self.observed_requests[-1].safety_settings[0].category,
            safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
        )

    def test_filters(self):
        self.responses["generate_text"] = glm.GenerateTextResponse(
            candidates=[{"output": "hello"}],
            filters=[
                {
                    "reason": safety_types.BlockedReason.SAFETY,
                    "message": "not safe",
                }
            ],
        )

        response = text_service.generate_text(prompt="do filters work?")
        self.assertIsInstance(response.filters[0]["reason"], safety_types.BlockedReason)
        self.assertEqual(response.filters[0]["reason"], safety_types.BlockedReason.SAFETY)

    def test_safety_feedback(self):
        self.responses["generate_text"] = glm.GenerateTextResponse(
            candidates=[{"output": "hello"}],
            safety_feedback=[
                {
                    "rating": {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                        "probability": safety_types.HarmProbability.HIGH,
                    },
                    "setting": {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                        "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                    },
                }
            ],
        )

        response = text_service.generate_text(prompt="does safety feedback work?")
        self.assertIsInstance(
            response.safety_feedback[0]["rating"]["probability"],
            safety_types.HarmProbability,
        )
        self.assertEqual(
            response.safety_feedback[0]["rating"]["probability"],
            safety_types.HarmProbability.HIGH,
        )

        self.assertIsInstance(
            response.safety_feedback[0]["setting"]["category"],
            safety_types.HarmCategory,
        )
        self.assertEqual(
            response.safety_feedback[0]["setting"]["category"],
            safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
        )

    def test_candidate_safety_feedback(self):
        self.responses["generate_text"] = glm.GenerateTextResponse(
            candidates=[
                {
                    "output": "hello",
                    "safety_ratings": [
                        {
                            "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                            "probability": safety_types.HarmProbability.HIGH,
                        },
                        {
                            "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                            "probability": safety_types.HarmProbability.LOW,
                        },
                    ],
                }
            ]
        )

        result = text_service.generate_text(prompt="Write a story from the ER.")
        self.assertIsInstance(
            result.candidates[0]["safety_ratings"][0]["category"],
            safety_types.HarmCategory,
        )
        self.assertEqual(
            result.candidates[0]["safety_ratings"][0]["category"],
            safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
        )

        self.assertIsInstance(
            result.candidates[0]["safety_ratings"][0]["probability"],
            safety_types.HarmProbability,
        )
        self.assertEqual(
            result.candidates[0]["safety_ratings"][0]["probability"],
            safety_types.HarmProbability.HIGH,
        )

    def test_candidate_citations(self):
        self.responses["generate_text"] = glm.GenerateTextResponse(
            candidates=[
                {
                    "output": "Hello Google!",
                    "citation_metadata": {
                        "citation_sources": [
                            {
                                "start_index": 6,
                                "end_index": 12,
                                "uri": "https://google.com",
                            }
                        ]
                    },
                }
            ]
        )
        result = text_service.generate_text(prompt="Hi my name is Google")
        self.assertEqual(
            result.candidates[0]["citation_metadata"]["citation_sources"][0]["start_index"],
            6,
        )

    @parameterized.named_parameters(
        [
            dict(testcase_name="base-name", model="models/text-bison-001"),
            dict(testcase_name="tuned-name", model="tunedModels/bipedal-pangolin-001"),
            dict(
                testcase_name="model",
                model=model_types.Model(
                    name="models/text-bison-001",
                    base_model_id="text-bison-001",
                    version="001",
                    display_name="🦬",
                    description="🦬🦬🦬🦬🦬🦬🦬🦬🦬🦬🦬",
                    input_token_limit=8000,
                    output_token_limit=4000,
                    supported_generation_methods=["GenerateText"],
                ),
            ),
            dict(
                testcase_name="tuned_model",
                model=model_types.TunedModel(
                    name="tunedModels/bipedal-pangolin-001",
                    base_model="models/text-bison-001",
                ),
            ),
            dict(
                testcase_name="glm_model",
                model=glm.Model(
                    name="models/text-bison-001",
                ),
            ),
            dict(
                testcase_name="glm_tuned_model",
                model=glm.TunedModel(
                    name="tunedModels/bipedal-pangolin-001",
                    base_model="models/text-bison-001",
                ),
            ),
            dict(
                testcase_name="glm_tuned_model_nested",
                model=glm.TunedModel(
                    name="tunedModels/bipedal-pangolin-002",
                    tuned_model_source={
                        "tuned_model": "tunedModels/bipedal-pangolin-002",
                        "base_model": "models/text-bison-001",
                    },
                ),
            ),
        ]
    )
    def test_count_message_tokens(self, model):
        self.responses["get_tuned_model"] = glm.TunedModel(
            name="tunedModels/bipedal-pangolin-001", base_model="models/text-bison-001"
        )
        self.responses["count_text_tokens"] = glm.CountTextTokensResponse(token_count=7)

        response = text_service.count_text_tokens(model, "Tell me a story about a magic backpack.")
        self.assertEqual({"token_count": 7}, response)

        should_look_up_model = isinstance(model, str) and model.startswith("tunedModels/")
        if should_look_up_model:
            self.assertLen(self.observed_requests, 2)
            self.assertEqual(
                self.observed_requests[0],
                glm.GetTunedModelRequest(name="tunedModels/bipedal-pangolin-001"),
            )

    def test_count_text_tokens_called_with_request_options(self):
        self.client.count_text_tokens = unittest.mock.MagicMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        try:
            result = text_service.count_text_tokens(
                model="models/",
                prompt="",
                request_options=request_options,
            )
        except AttributeError:
            pass

        self.client.count_text_tokens.assert_called_once_with(request, **request_options)

    def test_batch_embed_text_called_with_request_options(self):
        self.client.batch_embed_text = unittest.mock.MagicMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        try:
            result = text_service.generate_embeddings(
                model="models/",
                text=["first", "second"],
                request_options=request_options,
            )
        except AttributeError:
            pass

        self.client.batch_embed_text.assert_called_once_with(request, **request_options)

    def test_embed_text_called_with_request_options(self):
        self.client.embed_text = unittest.mock.MagicMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        try:
            result = text_service.generate_embeddings(
                model="models/",
                text="",
                request_options=request_options,
            )
        except AttributeError:
            pass

        self.client.embed_text.assert_called_once_with(request, **request_options)

    def test_generate_text_called_with_request_options(self):
        self.client.generate_text = unittest.mock.MagicMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        try:
            result = text_service.generate_text(prompt="", request_options=request_options)
        except AttributeError:
            pass

        self.client.generate_text.assert_called_once_with(request, **request_options)


if __name__ == "__main__":
    absltest.main()
