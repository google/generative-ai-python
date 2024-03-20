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

from google.generativeai import answer
from google.generativeai import client
from absl.testing import absltest
from absl.testing import parameterized

from google.generativeai.types import content_types

DEFAULT_ANSWER_MODEL = "models/aqa"


class UnitTests(parameterized.TestCase):
    def setUp(self):
        self.client = unittest.mock.MagicMock()

        client._client_manager.clients["generative"] = self.client
        client._client_manager.clients["model"] = self.client

        self.observed_requests = []

        def add_client_method(f):
            name = f.__name__
            setattr(self.client, name, f)
            return f

        @add_client_method
        def generate_answer(
            request: glm.GenerateAnswerRequest,
            **kwargs,
        ) -> glm.GenerateAnswerResponse:
            self.observed_requests.append(request)
            return glm.GenerateAnswerResponse(
                answer=glm.Candidate(
                    index=1,
                    content=(glm.Content(parts=[glm.Part(text="Demo answer.")])),
                ),
                answerable_probability=0.500,
            )

    def test_make_grounding_passages_mixed_types(self):
        inline_passages = [
            "I am a chicken",
            glm.Content(parts=[glm.Part(text="I am a bird.")]),
            glm.Content(parts=[glm.Part(text="I can fly!")]),
        ]
        x = answer._make_grounding_passages(inline_passages)
        self.assertIsInstance(x, glm.GroundingPassages)
        self.assertEqual(
            glm.GroundingPassages(
                passages=[
                    {"id": "0", "content": glm.Content(parts=[glm.Part(text="I am a chicken")])},
                    {"id": "1", "content": glm.Content(parts=[glm.Part(text="I am a bird.")])},
                    {"id": "2", "content": glm.Content(parts=[glm.Part(text="I can fly!")])},
                ]
            ),
            x,
        )

    @parameterized.named_parameters(
        [
            dict(
                testcase_name="grounding_passage",
                inline_passages=glm.GroundingPassages(
                    passages=[
                        {
                            "id": "0",
                            "content": glm.Content(parts=[glm.Part(text="I am a chicken")]),
                        },
                        {"id": "1", "content": glm.Content(parts=[glm.Part(text="I am a bird.")])},
                        {"id": "2", "content": glm.Content(parts=[glm.Part(text="I can fly!")])},
                    ]
                ),
            ),
            dict(
                testcase_name="content_object",
                inline_passages=[
                    glm.Content(parts=[glm.Part(text="I am a chicken")]),
                    glm.Content(parts=[glm.Part(text="I am a bird.")]),
                    glm.Content(parts=[glm.Part(text="I can fly!")]),
                ],
            ),
            dict(
                testcase_name="list_of_strings",
                inline_passages=["I am a chicken", "I am a bird.", "I can fly!"],
            ),
        ]
    )
    def test_make_grounding_passages(self, inline_passages):
        x = answer._make_grounding_passages(inline_passages)
        self.assertIsInstance(x, glm.GroundingPassages)
        self.assertEqual(
            glm.GroundingPassages(
                passages=[
                    {"id": "0", "content": glm.Content(parts=[glm.Part(text="I am a chicken")])},
                    {"id": "1", "content": glm.Content(parts=[glm.Part(text="I am a bird.")])},
                    {"id": "2", "content": glm.Content(parts=[glm.Part(text="I can fly!")])},
                ]
            ),
            x,
        )

    @parameterized.named_parameters(
        dict(
            testcase_name="dict_of_strings",
            inline_passages={"4": "I am a chicken", "5": "I am a bird.", "6": "I can fly!"},
        ),
        dict(
            testcase_name="tuple_of_strings",
            inline_passages=[("4", "I am a chicken"), ("5", "I am a bird."), ("6", "I can fly!")],
        ),
        dict(
            testcase_name="list_of_grounding_passages",
            inline_passages=[
                glm.GroundingPassage(
                    id="4", content=glm.Content(parts=[glm.Part(text="I am a chicken")])
                ),
                glm.GroundingPassage(
                    id="5", content=glm.Content(parts=[glm.Part(text="I am a bird.")])
                ),
                glm.GroundingPassage(
                    id="6", content=glm.Content(parts=[glm.Part(text="I can fly!")])
                ),
            ],
        ),
    )
    def test_make_grounding_passages_different_id(self, inline_passages):
        x = answer._make_grounding_passages(inline_passages)
        self.assertIsInstance(x, glm.GroundingPassages)
        self.assertEqual(
            glm.GroundingPassages(
                passages=[
                    {"id": "4", "content": glm.Content(parts=[glm.Part(text="I am a chicken")])},
                    {"id": "5", "content": glm.Content(parts=[glm.Part(text="I am a bird.")])},
                    {"id": "6", "content": glm.Content(parts=[glm.Part(text="I can fly!")])},
                ]
            ),
            x,
        )

    def test_make_grounding_passages_key_strings(self):
        inline_passages = {
            "first": "I am a chicken",
            "second": "I am a bird.",
            "third": "I can fly!",
        }

        x = answer._make_grounding_passages(inline_passages)
        self.assertIsInstance(x, glm.GroundingPassages)
        self.assertEqual(
            glm.GroundingPassages(
                passages=[
                    {
                        "id": "first",
                        "content": glm.Content(parts=[glm.Part(text="I am a chicken")]),
                    },
                    {"id": "second", "content": glm.Content(parts=[glm.Part(text="I am a bird.")])},
                    {"id": "third", "content": glm.Content(parts=[glm.Part(text="I can fly!")])},
                ]
            ),
            x,
        )

    def test_generate_answer_request(self):
        # Should be a list of contents to use to_contents() function.
        contents = [glm.Content(parts=[glm.Part(text="I have wings.")])]

        inline_passages = ["I am a chicken", "I am a bird.", "I can fly!"]
        grounding_passages = glm.GroundingPassages(
            passages=[
                {"id": "0", "content": glm.Content(parts=[glm.Part(text="I am a chicken")])},
                {"id": "1", "content": glm.Content(parts=[glm.Part(text="I am a bird.")])},
                {"id": "2", "content": glm.Content(parts=[glm.Part(text="I can fly!")])},
            ]
        )

        x = answer._make_generate_answer_request(
            model=DEFAULT_ANSWER_MODEL, contents=contents, grounding_source=inline_passages
        )

        self.assertEqual(
            glm.GenerateAnswerRequest(
                model=DEFAULT_ANSWER_MODEL, contents=contents, inline_passages=grounding_passages
            ),
            x,
        )

    def test_generate_answer_called_with_request_options(self):
        self.client.generate_answer = mock.MagicMock()
        request = mock.ANY
        request_options = {"timeout": 120}

        answer.generate_answer(contents=[], inline_passages=[], request_options=request_options)

        self.client.generate_answer.assert_called_once_with(request, **request_options)


if __name__ == "__main__":
    absltest.main()
