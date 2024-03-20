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
from typing import Any

import unittest.mock

import google.ai.generativelanguage as glm

from google.generativeai import discuss
from google.generativeai import client
import google.generativeai as genai
from google.generativeai.types import safety_types

from absl.testing import absltest
from absl.testing import parameterized

# TODO: replace returns with 'assert' statements


class UnitTests(parameterized.TestCase):
    def setUp(self):
        self.client = unittest.mock.MagicMock()

        client._client_manager.clients["discuss"] = self.client

        self.observed_request = None

        self.mock_response = glm.GenerateMessageResponse(
            candidates=[
                glm.Message(content="a", author="1"),
                glm.Message(content="b", author="1"),
                glm.Message(content="c", author="1"),
            ],
        )

        def fake_generate_message(
            request: glm.GenerateMessageRequest,
            **kwargs,
        ) -> glm.GenerateMessageResponse:
            self.observed_request = request
            response = copy.copy(self.mock_response)
            response.messages = request.prompt.messages
            return response

        self.client.generate_message = fake_generate_message

    @parameterized.named_parameters(
        ["string", "Hello", ""],
        ["dict", {"content": "Hello"}, ""],
        ["dict_author", {"content": "Hello", "author": "me"}, "me"],
        ["proto", glm.Message(content="Hello"), ""],
        ["proto_author", glm.Message(content="Hello", author="me"), "me"],
    )
    def test_make_message(self, message, author):
        x = discuss._make_message(message)
        self.assertIsInstance(x, glm.Message)
        self.assertEqual("Hello", x.content)
        self.assertEqual(author, x.author)

    @parameterized.named_parameters(
        ["string", "Hello", ["Hello"]],
        ["dict", {"content": "Hello"}, ["Hello"]],
        ["proto", glm.Message(content="Hello"), ["Hello"]],
        [
            "list",
            ["hello0", {"content": "hello1"}, glm.Message(content="hello2")],
            ["hello0", "hello1", "hello2"],
        ],
    )
    def test_make_messages(self, messages, expected_contents):
        messages = discuss._make_messages(messages)
        for expected, message in zip(expected_contents, messages):
            self.assertEqual(expected, message.content)

    @parameterized.named_parameters(
        ["tuple", ("hello", {"content": "goodbye"})],
        ["iterable", iter(["hello", "goodbye"])],
        ["dict", {"input": "hello", "output": "goodbye"}],
        [
            "proto",
            glm.Example(
                input=glm.Message(content="hello"),
                output=glm.Message(content="goodbye"),
            ),
        ],
    )
    def test_make_example(self, example):
        x = discuss._make_example(example)
        self.assertIsInstance(x, glm.Example)
        self.assertEqual("hello", x.input.content)
        self.assertEqual("goodbye", x.output.content)
        return

    @parameterized.named_parameters(
        [
            "messages",
            [
                "Hi",
                {"content": "Hello!"},
                "what's your name?",
                glm.Message(content="Dave, what's yours"),
            ],
        ],
        [
            "examples",
            [
                ("Hi", "Hello!"),
                {
                    "input": "what's your name?",
                    "output": {"content": "Dave, what's yours"},
                },
            ],
        ],
    )
    def test_make_examples(self, examples):
        examples = discuss._make_examples(examples)
        self.assertLen(examples, 2)
        self.assertEqual(examples[0].input.content, "Hi")
        self.assertEqual(examples[0].output.content, "Hello!")
        self.assertEqual(examples[1].input.content, "what's your name?")
        self.assertEqual(examples[1].output.content, "Dave, what's yours")

        return

    def test_make_examples_from_example(self):
        ex_dict = {"input": "hello", "output": "meow!"}
        example = discuss._make_example(ex_dict)
        examples1 = discuss._make_examples(ex_dict)
        examples2 = discuss._make_examples(discuss._make_example(ex_dict))

        self.assertEqual(example, examples1[0])
        self.assertEqual(example, examples2[0])

    @parameterized.named_parameters(
        ["str", "hello"],
        ["message", glm.Message(content="hello")],
        ["messages", ["hello"]],
        ["dict", {"messages": "hello"}],
        ["dict2", {"messages": ["hello"]}],
        ["proto", glm.MessagePrompt(messages=[glm.Message(content="hello")])],
    )
    def test_make_message_prompt_from_messages(self, prompt):
        x = discuss._make_message_prompt(prompt)
        self.assertIsInstance(x, glm.MessagePrompt)
        self.assertEqual(x.messages[0].content, "hello")
        return

    @parameterized.named_parameters(
        [
            "dict",
            [
                {
                    "context": "you are a cat",
                    "examples": ["are you hungry?", "meow!"],
                    "messages": "hello",
                }
            ],
            {},
        ],
        [
            "kwargs",
            [],
            {
                "context": "you are a cat",
                "examples": ["are you hungry?", "meow!"],
                "messages": "hello",
            },
        ],
        [
            "proto",
            [
                glm.MessagePrompt(
                    context="you are a cat",
                    examples=[
                        glm.Example(
                            input=glm.Message(content="are you hungry?"),
                            output=glm.Message(content="meow!"),
                        )
                    ],
                    messages=[glm.Message(content="hello")],
                )
            ],
            {},
        ],
    )
    def test_make_message_prompt_from_prompt(self, args, kwargs):
        x = discuss._make_message_prompt(*args, **kwargs)
        self.assertIsInstance(x, glm.MessagePrompt)
        self.assertEqual(x.context, "you are a cat")
        self.assertEqual(x.examples[0].input.content, "are you hungry?")
        self.assertEqual(x.examples[0].output.content, "meow!")
        self.assertEqual(x.messages[0].content, "hello")

    def test_make_generate_message_request_nested(
        self,
    ):
        request0 = discuss._make_generate_message_request(
            **{
                "model": "models/Dave",
                "context": "you are a cat",
                "examples": ["hello", "meow", "are you hungry?", "meow!"],
                "messages": "Please catch that mouse.",
                "temperature": 0.2,
                "candidate_count": 7,
            }
        )
        request1 = discuss._make_generate_message_request(
            **{
                "model": "models/Dave",
                "prompt": {
                    "context": "you are a cat",
                    "examples": ["hello", "meow", "are you hungry?", "meow!"],
                    "messages": "Please catch that mouse.",
                },
                "temperature": 0.2,
                "candidate_count": 7,
            }
        )

        self.assertIsInstance(request0, glm.GenerateMessageRequest)
        self.assertIsInstance(request1, glm.GenerateMessageRequest)
        self.assertEqual(request0, request1)

    @parameterized.parameters(
        {"prompt": {}, "context": "You are a cat."},
        {
            "prompt": {"context": "You are a cat."},
            "examples": ["hello", "meow"],
        },
        {"prompt": {"examples": ["hello", "meow"]}, "messages": "hello"},
    )
    def test_make_generate_message_request_flat_prompt_conflict(
        self,
        context=None,
        examples=None,
        messages=None,
        prompt=None,
    ):
        with self.assertRaises(ValueError):
            x = discuss._make_generate_message_request(
                model="test",
                context=context,
                examples=examples,
                messages=messages,
                prompt=prompt,
            )

    @parameterized.parameters(
        {"kwargs": {"context": "You are a cat."}},
        {"kwargs": {"messages": "hello"}},
        {"kwargs": {"examples": [["a", "b"], ["c", "d"]]}},
        {
            "kwargs": {
                "messages": ["hello"],
                "examples": [["a", "b"], ["c", "d"]],
            }
        },
    )
    def test_reply(self, kwargs):
        response = genai.chat(**kwargs)
        first_messages = response.messages

        self.assertEqual("a", response.last)
        self.assertEqual(
            [
                {"author": "1", "content": "a"},
                {"author": "1", "content": "b"},
                {"author": "1", "content": "c"},
            ],
            response.candidates,
        )

        response = response.reply("again")

    def test_receive_and_reply_with_filters(self):
        self.mock_response = mock_response = glm.GenerateMessageResponse(
            candidates=[glm.Message(content="a", author="1")],
            filters=[
                glm.ContentFilter(reason=safety_types.BlockedReason.SAFETY, message="unsafe"),
                glm.ContentFilter(reason=safety_types.BlockedReason.OTHER),
            ],
        )
        response = discuss.chat(messages="do filters work?")

        filters = response.filters
        self.assertLen(filters, 2)
        self.assertIsInstance(filters[0]["reason"], safety_types.BlockedReason)
        self.assertEqual(filters[0]["reason"], safety_types.BlockedReason.SAFETY)
        self.assertEqual(filters[0]["message"], "unsafe")

        self.mock_response = glm.GenerateMessageResponse(
            candidates=[glm.Message(content="a", author="1")],
            filters=[
                glm.ContentFilter(reason=safety_types.BlockedReason.BLOCKED_REASON_UNSPECIFIED)
            ],
        )

        response = response.reply("Does reply work?")
        filters = response.filters
        self.assertLen(filters, 1)
        self.assertIsInstance(filters[0]["reason"], safety_types.BlockedReason)
        self.assertEqual(
            filters[0]["reason"],
            safety_types.BlockedReason.BLOCKED_REASON_UNSPECIFIED,
        )

    def test_chat_citations(self):
        self.mock_response = mock_response = glm.GenerateMessageResponse(
            candidates=[
                {
                    "content": "Hello google!",
                    "author": "1",
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
            ],
        )

        response = discuss.chat(messages="Do citations work?")

        self.assertEqual(
            response.candidates[0]["citation_metadata"]["citation_sources"][0]["start_index"],
            6,
        )

        response = response.reply("What about a second time?")

        self.assertEqual(
            response.candidates[0]["citation_metadata"]["citation_sources"][0]["start_index"],
            6,
        )
        self.assertLen(response.messages, 4)

    def test_set_last(self):
        response = discuss.chat(messages="Can you overwrite `.last`?")
        response.last = "yes"
        response = response.reply("glad to hear it!")
        response.last = "Me too!"
        self.assertEqual(
            [msg["content"] for msg in response.messages],
            [
                "Can you overwrite `.last`?",
                "yes",
                "glad to hear it!",
                "Me too!",
            ],
        )

    def test_generate_message_called_with_request_options(self):
        self.client.generate_message = unittest.mock.MagicMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        try:
            genai.chat(**{"context": "You are a cat."}, request_options=request_options)
        except AttributeError:
            pass

        self.client.generate_message.assert_called_once_with(request, **request_options)


if __name__ == "__main__":
    absltest.main()
