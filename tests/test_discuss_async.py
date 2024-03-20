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

import sys
from typing import Any
import unittest

import google.ai.generativelanguage as glm

from google.generativeai import discuss
from absl.testing import absltest
from absl.testing import parameterized


class AsyncTests(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):
    async def test_chat_async(self):
        client = unittest.mock.AsyncMock()

        observed_request = None

        async def fake_generate_message(
            request: glm.GenerateMessageRequest,
            **kwargs,
        ) -> glm.GenerateMessageResponse:
            nonlocal observed_request
            observed_request = request
            return glm.GenerateMessageResponse(
                candidates=[
                    glm.Message(
                        author="1",
                        content="Why did the chicken cross the road?",
                    )
                ]
            )

        client.generate_message = fake_generate_message

        observed_response = await discuss.chat_async(
            model="models/bard",
            context="Example Prompt",
            examples=[["Example from human", "Example response from AI"]],
            messages=["Tell me a joke"],
            temperature=0.75,
            candidate_count=1,
            client=client,
        )

        self.assertEqual(
            observed_request,
            glm.GenerateMessageRequest(
                model="models/bard",
                prompt=glm.MessagePrompt(
                    context="Example Prompt",
                    examples=[
                        glm.Example(
                            input=glm.Message(content="Example from human"),
                            output=glm.Message(content="Example response from AI"),
                        )
                    ],
                    messages=[glm.Message(author="0", content="Tell me a joke")],
                ),
                temperature=0.75,
                candidate_count=1,
            ),
        )
        self.assertEqual(
            observed_response.candidates,
            [{"author": "1", "content": "Why did the chicken cross the road?"}],
        )


if __name__ == "__main__":
    absltest.main()
