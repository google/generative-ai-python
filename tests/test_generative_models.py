import collections
from collections.abc import Iterable
import copy
import pathlib
from typing import Any
import textwrap
import unittest.mock
from absl.testing import absltest
from absl.testing import parameterized
import google.ai.generativelanguage as glm
from google.generativeai import client as client_lib
from google.generativeai import generative_models
from google.generativeai.types import content_types
from google.generativeai.types import generation_types

import PIL.Image

HERE = pathlib.Path(__file__).parent
TEST_IMAGE_PATH = HERE / "test_img.png"
TEST_IMAGE_URL = "https://storage.googleapis.com/generativeai-downloads/data/test_img.png"
TEST_IMAGE_DATA = TEST_IMAGE_PATH.read_bytes()


def simple_response(text: str) -> glm.GenerateContentResponse:
    return glm.GenerateContentResponse({"candidates": [{"content": {"parts": [{"text": text}]}}]})


class CUJTests(parameterized.TestCase):
    """Tests are in order with the design doc."""

    def setUp(self):
        self.client = unittest.mock.MagicMock()

        client_lib._client_manager.clients["generative"] = self.client

        def add_client_method(f):
            name = f.__name__
            setattr(self.client, name, f)
            return f

        self.observed_requests = []
        self.responses = collections.defaultdict(list)

        @add_client_method
        def generate_content(
            request: glm.GenerateContentRequest,
            **kwargs,
        ) -> glm.GenerateContentResponse:
            self.assertIsInstance(request, glm.GenerateContentRequest)
            self.observed_requests.append(request)
            response = self.responses["generate_content"].pop(0)
            return response

        @add_client_method
        def stream_generate_content(
            request: glm.GetModelRequest,
            **kwargs,
        ) -> Iterable[glm.GenerateContentResponse]:
            self.observed_requests.append(request)
            response = self.responses["stream_generate_content"].pop(0)
            return response

        @add_client_method
        def count_tokens(
            request: glm.CountTokensRequest,
            **kwargs,
        ) -> Iterable[glm.GenerateContentResponse]:
            self.observed_requests.append(request)
            response = self.responses["count_tokens"].pop(0)
            return response

    def test_hello(self):
        # Generate text from text prompt
        model = generative_models.GenerativeModel(model_name="gemini-pro")

        self.responses["generate_content"].append(simple_response("world!"))

        response = model.generate_content("Hello")

        self.assertEqual(self.observed_requests[0].contents[0].parts[0].text, "Hello")
        self.assertEqual(response.candidates[0].content.parts[0].text, "world!")

        self.assertEqual(response.text, "world!")

    @parameterized.named_parameters(
        [
            "JustImage",
            PIL.Image.open(TEST_IMAGE_PATH),
        ],
        [
            "ImageAndText",
            ["What's in this picture?", PIL.Image.open(TEST_IMAGE_PATH)],
        ],
    )
    def test_image(self, content):
        # Generate text from image
        model = generative_models.GenerativeModel("gemini-pro")

        cat = "It's a cat"
        self.responses["generate_content"].append(simple_response(cat))

        response = model.generate_content(content)

        self.assertEqual(
            self.observed_requests[0].contents[0].parts[-1].inline_data.mime_type,
            "image/png",
        )
        self.assertEqual(
            self.observed_requests[0].contents[0].parts[-1].inline_data.data[:4],
            b"\x89PNG",
        )
        self.assertEqual(response.candidates[0].content.parts[0].text, cat)

        self.assertEqual(response.text, cat)

    @parameterized.named_parameters(
        ["dict", {"temperature": 0.0}, {"temperature": 0.5}],
        [
            "object",
            generation_types.GenerationConfig(temperature=0.0),
            generation_types.GenerationConfig(temperature=0.5),
        ],
        [
            "glm",
            glm.GenerationConfig(temperature=0.0),
            glm.GenerationConfig(temperature=0.5),
        ],
    )
    def test_generation_config_overwrite(self, config1, config2):
        # Generation config
        model = generative_models.GenerativeModel("gemini-pro", generation_config=config1)

        self.responses["generate_content"] = [
            simple_response(" world!"),
            simple_response(" world!"),
        ]

        _ = model.generate_content("hello")
        self.assertEqual(self.observed_requests[-1].generation_config.temperature, 0.0)

        _ = model.generate_content("hello", generation_config=config2)
        self.assertEqual(self.observed_requests[-1].generation_config.temperature, 0.5)

    @parameterized.named_parameters(
        ["dict", {"danger": "low"}, {"danger": "high"}],
        [
            "list-dict",
            [
                dict(
                    category=glm.HarmCategory.HARM_CATEGORY_DANGEROUS,
                    threshold=glm.SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                ),
            ],
            [
                dict(category="danger", threshold="high"),
            ],
        ],
        [
            "object",
            [
                glm.SafetySetting(
                    category=glm.HarmCategory.HARM_CATEGORY_DANGEROUS,
                    threshold=glm.SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                ),
            ],
            [
                glm.SafetySetting(
                    category=glm.HarmCategory.HARM_CATEGORY_DANGEROUS,
                    threshold=glm.SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                ),
            ],
        ],
    )
    def test_safety_overwrite(self, safe1, safe2):
        # Safety
        model = generative_models.GenerativeModel("gemini-pro", safety_settings={"danger": "low"})

        self.responses["generate_content"] = [
            simple_response(" world!"),
            simple_response(" world!"),
        ]

        _ = model.generate_content("hello")
        self.assertEqual(
            self.observed_requests[-1].safety_settings[0].category,
            glm.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )
        self.assertEqual(
            self.observed_requests[-1].safety_settings[0].threshold,
            glm.SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        )

        _ = model.generate_content("hello", safety_settings={"danger": "high"})
        self.assertEqual(
            self.observed_requests[-1].safety_settings[0].category,
            glm.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )
        self.assertEqual(
            self.observed_requests[-1].safety_settings[0].threshold,
            glm.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        )

    def test_stream_basic(self):
        # Streaming
        chunks = ["first", " second", " third"]
        self.responses["stream_generate_content"] = [(simple_response(text) for text in chunks)]

        model = generative_models.GenerativeModel("gemini-pro")
        response = model.generate_content("Hello", stream=True)

        self.assertEqual(self.observed_requests[0].contents[0].parts[0].text, "Hello")

        for n, got in enumerate(response):
            self.assertEqual(chunks[n], got.text)

        self.assertEqual(response.text, "".join(chunks))

    def test_stream_lookahead(self):
        chunks = ["first", " second", " third"]
        self.responses["stream_generate_content"] = [(simple_response(text) for text in chunks)]

        model = generative_models.GenerativeModel("gemini-pro")
        response = model.generate_content("Hello", stream=True)

        self.assertEqual(self.observed_requests[0].contents[0].parts[0].text, "Hello")

        for expected, got in zip(chunks, response):
            self.assertEqual(expected, got.text)

        self.assertEqual(response.text, "".join(chunks))

    def test_stream_prompt_feedback_blocked(self):
        chunks = [
            glm.GenerateContentResponse(
                {
                    "prompt_feedback": {"block_reason": "SAFETY"},
                }
            )
        ]
        self.responses["stream_generate_content"] = [(chunk for chunk in chunks)]

        model = generative_models.GenerativeModel("gemini-pro")
        response = model.generate_content("Bad stuff!", stream=True)

        self.assertEqual(
            response.prompt_feedback.block_reason,
            glm.GenerateContentResponse.PromptFeedback.BlockReason.SAFETY,
        )

        with self.assertRaises(generation_types.BlockedPromptException):
            for chunk in response:
                pass

    def test_stream_prompt_feedback_not_blocked(self):
        chunks = [
            glm.GenerateContentResponse(
                {
                    "prompt_feedback": {
                        "safety_ratings": [
                            {
                                "category": glm.HarmCategory.HARM_CATEGORY_DANGEROUS,
                                "probability": glm.SafetyRating.HarmProbability.NEGLIGIBLE,
                            }
                        ]
                    },
                    "candidates": [{"content": {"parts": [{"text": "first"}]}}],
                }
            ),
            glm.GenerateContentResponse(
                {
                    "candidates": [{"content": {"parts": [{"text": " second"}]}}],
                }
            ),
        ]
        self.responses["stream_generate_content"] = [(chunk for chunk in chunks)]

        model = generative_models.GenerativeModel("gemini-pro")
        response = model.generate_content("Hello", stream=True)

        self.assertEqual(
            response.prompt_feedback.safety_ratings[0].category,
            glm.HarmCategory.HARM_CATEGORY_DANGEROUS,
        )

        text = "".join(chunk.text for chunk in response)
        self.assertEqual(text, "first second")

    def test_chat(self):
        # Multi turn chat
        model = generative_models.GenerativeModel("gemini-pro")
        chat = model.start_chat()

        self.responses["generate_content"] = [
            simple_response("first"),
            simple_response("second"),
            simple_response("third"),
        ]

        msg1 = "I really like fantasy books."
        response = chat.send_message(msg1)
        self.assertEqual(response.text, "first")
        self.assertEqual(self.observed_requests[0].contents[0].parts[0].text, msg1)

        msg2 = "I also like this image."
        response = chat.send_message([msg2, PIL.Image.open(TEST_IMAGE_PATH)])

        self.assertEqual(response.text, "second")
        self.assertEqual(self.observed_requests[1].contents[0].parts[0].text, msg1)
        self.assertEqual(self.observed_requests[1].contents[1].parts[0].text, "first")
        self.assertEqual(self.observed_requests[1].contents[2].parts[0].text, msg2)
        self.assertEqual(
            self.observed_requests[1].contents[2].parts[1].inline_data.data[:4],
            b"\x89PNG",
        )

        msg3 = "What things do I like?."
        response = chat.send_message(msg3)
        self.assertEqual(response.text, "third")
        self.assertLen(chat.history, 6)

    def test_chat_roles(self):
        self.responses["generate_content"] = [simple_response("hello!")]

        model = generative_models.GenerativeModel("gemini-pro")
        chat = model.start_chat()
        response = chat.send_message("hello?")
        history = chat.history
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1].role, "model")

    def test_chat_streaming_basic(self):
        # Chat streaming
        self.responses["stream_generate_content"] = [
            iter([simple_response("a"), simple_response("b"), simple_response("c")]),
            iter([simple_response("1"), simple_response("2"), simple_response("3")]),
            iter([simple_response("x"), simple_response("y"), simple_response("z")]),
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()

        response = chat.send_message("letters?", stream=True)

        self.assertEqual("".join(chunk.text for chunk in response), "abc")

        response = chat.send_message("numbers?", stream=True)

        self.assertEqual("".join(chunk.text for chunk in response), "123")

        response = chat.send_message("more letters?", stream=True)

        self.assertEqual("".join(chunk.text for chunk in response), "xyz")

    def test_chat_incomplete_streaming_errors(self):
        # Chat streaming
        self.responses["stream_generate_content"] = [
            iter([simple_response("a"), simple_response("b"), simple_response("c")]),
            iter([simple_response("1"), simple_response("2"), simple_response("3")]),
            iter([simple_response("x"), simple_response("y"), simple_response("z")]),
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()
        response = chat.send_message("letters?", stream=True)

        with self.assertRaises(generation_types.IncompleteIterationError):
            chat.history

        with self.assertRaises(generation_types.IncompleteIterationError):
            chat.send_message("numbers?", stream=True)

        for chunk in response:
            pass
        self.assertLen(chat.history, 2)

        response = chat.send_message("numbers?", stream=True)
        self.assertEqual("".join(chunk.text for chunk in response), "123")

    def test_edit_history(self):
        self.responses["generate_content"] = [
            simple_response("first"),
            simple_response("second"),
            simple_response("third"),
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()

        response = chat.send_message("hello")
        self.assertEqual(response.text, "first")
        self.assertLen(chat.history, 2)

        response = chat.send_message("hello")
        self.assertEqual(response.text, "second")
        self.assertLen(chat.history, 4)

        chat.history[-1] = content_types.to_content("edited")
        response = chat.send_message("hello")
        self.assertEqual(response.text, "third")
        self.assertLen(chat.history, 6)

        self.assertEqual(chat.history[3], content_types.to_content("edited"))
        self.assertEqual(self.observed_requests[-1].contents[3].parts[0].text, "edited")

    def test_replace_history(self):
        self.responses["generate_content"] = [
            simple_response("first"),
            simple_response("second"),
            simple_response("third"),
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()
        chat.send_message("hello1")
        chat.send_message("hello2")

        self.assertLen(chat.history, 4)
        chat.history = [{"parts": ["Goodbye"]}, {"parts": ["Later gater"]}]
        self.assertLen(chat.history, 2)

        response = chat.send_message("hello3")
        self.assertEqual(response.text, "third")
        self.assertLen(chat.history, 4)

        self.assertEqual(self.observed_requests[-1].contents[0].parts[0].text, "Goodbye")

    def test_copy_history(self):
        self.responses["generate_content"] = [
            simple_response("first"),
            simple_response("second"),
            simple_response("third"),
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat1 = model.start_chat()
        chat1.send_message("hello1")

        chat2 = copy.deepcopy(chat1)
        chat2.send_message("hello2")

        chat1.send_message("hello3")

        self.assertLen(chat1.history, 4)
        expected = [
            {"role": "user", "parts": ["hello1"]},
            {"role": "model", "parts": ["first"]},
            {"role": "user", "parts": ["hello3"]},
            {"role": "model", "parts": ["third"]},
        ]
        for content, ex in zip(chat1.history, expected):
            self.assertEqual(content, content_types.to_content(ex))

        self.assertLen(chat2.history, 4)
        expected = [
            {"role": "user", "parts": ["hello1"]},
            {"role": "model", "parts": ["first"]},
            {"role": "user", "parts": ["hello2"]},
            {"role": "model", "parts": ["second"]},
        ]
        for content, ex in zip(chat2.history, expected):
            self.assertEqual(content, content_types.to_content(ex))

    def test_chat_error_in_stream(self):
        def throw():
            for c in "123":
                yield simple_response(c)
            raise ValueError()

        def no_throw():
            for c in "abc":
                yield simple_response(c)

        self.responses["stream_generate_content"] = [
            no_throw(),
            throw(),
            no_throw(),
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()

        # Send a message, the response is okay..
        chat.send_message("hello1", stream=True).resolve()

        # Send a second message, it fails
        response = chat.send_message("hello2", stream=True)
        with self.assertRaises(ValueError):
            # Iteration fails
            for chunk in response:
                pass

        # Since the response broke, we can't access the history
        with self.assertRaises(generation_types.BrokenResponseError):
            chat.history

        # or send another message.
        with self.assertRaises(generation_types.BrokenResponseError):
            chat.send_message("hello")

        # Rewind a step to before the error
        chat.rewind()

        self.assertLen(chat.history, 2)
        self.assertEqual(chat.history[0].parts[0].text, "hello1")
        self.assertEqual(chat.history[1].parts[0].text, "abc")

        # And continue
        chat.send_message("hello3", stream=True).resolve()
        self.assertLen(chat.history, 4)
        self.assertEqual(chat.history[2].parts[0].text, "hello3")
        self.assertEqual(chat.history[3].parts[0].text, "abc")

    def test_chat_prompt_blocked(self):
        self.responses["generate_content"] = [
            glm.GenerateContentResponse(
                {
                    "prompt_feedback": {"block_reason": "SAFETY"},
                }
            )
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()

        with self.assertRaises(generation_types.BlockedPromptException):
            chat.send_message("hello")

        self.assertLen(chat.history, 0)

    def test_chat_candidate_blocked(self):
        # I feel like chat needs a .last so you can look at the partial results.
        self.responses["generate_content"] = [
            glm.GenerateContentResponse(
                {
                    "candidates": [{"finish_reason": "SAFETY"}],
                }
            )
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()

        with self.assertRaises(generation_types.StopCandidateException):
            chat.send_message("hello")

    def test_chat_streaming_unexpected_stop(self):
        self.responses["stream_generate_content"] = [
            iter(
                [
                    simple_response("a"),
                    simple_response("b"),
                    simple_response("c"),
                    glm.GenerateContentResponse(
                        {
                            "candidates": [{"finish_reason": "SAFETY"}],
                        }
                    ),
                ]
            )
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()

        response = chat.send_message("hello", stream=True)
        for chunk in response:
            # The result doesn't know it's a chat result so it can't throw.
            # Unless we give it some way to know?
            pass

        with self.assertRaises(generation_types.BrokenResponseError):
            # But when preparing the next message, we can throw:
            response = chat.send_message("hello2", stream=True)

        # It's a little bad that here it's only on send message that you find out
        # about the problem. "hello2" is never added, the `rewind` removes `hello1`.
        chat.rewind()
        self.assertLen(chat.history, 0)

    def test_tools(self):
        tools = dict(
            function_declarations=[
                dict(name="datetime", description="Returns the current UTC date and time.")
            ]
        )
        model = generative_models.GenerativeModel("gemini-pro-vision", tools=tools)

        self.responses["generate_content"] = [
            simple_response("a"),
            simple_response("b"),
        ]

        response = model.generate_content("Hello")

        chat = model.start_chat()
        response = chat.send_message("Hello")

        expect_tools = dict(
            function_declarations=[
                dict(name="datetime", description="Returns the current UTC date and time.")
            ]
        )

        for obr in self.observed_requests:
            self.assertLen(obr.tools, 1)
            self.assertEqual(type(obr.tools[0]).to_dict(obr.tools[0]), tools)

    @parameterized.named_parameters(
        ["basic", "Hello"],
        ["list", ["Hello"]],
        [
            "list2",
            [{"text": "Hello"}, {"inline_data": {"data": b"PNG!", "mime_type": "image/png"}}],
        ],
        ["contents", [{"role": "user", "parts": ["hello"]}]],
    )
    def test_count_tokens_smoke(self, contents):
        self.responses["count_tokens"] = [glm.CountTokensResponse(total_tokens=7)]
        model = generative_models.GenerativeModel("gemini-pro-vision")
        response = model.count_tokens(contents)
        self.assertEqual(type(response).to_dict(response), {"total_tokens": 7})

    @parameterized.named_parameters(
        [
            "GenerateContentResponse",
            generation_types.GenerateContentResponse,
            generation_types.AsyncGenerateContentResponse,
        ],
        [
            "GenerativeModel.generate_response",
            generative_models.GenerativeModel.generate_content,
            generative_models.GenerativeModel.generate_content_async,
        ],
        [
            "GenerativeModel.count_tokens",
            generative_models.GenerativeModel.count_tokens,
            generative_models.GenerativeModel.count_tokens_async,
        ],
        [
            "ChatSession.send_message",
            generative_models.ChatSession.send_message,
            generative_models.ChatSession.send_message_async,
        ],
        [
            "ChatSession._handle_afc",
            generative_models.ChatSession._handle_afc,
            generative_models.ChatSession._handle_afc_async,
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

    def test_repr_for_unary_non_streamed_response(self):
        model = generative_models.GenerativeModel(model_name="gemini-pro")
        self.responses["generate_content"].append(simple_response("world!"))
        response = model.generate_content("Hello")

        result = repr(response)
        expected = textwrap.dedent(
            """\
            response:
            GenerateContentResponse(
                done=True,
                iterator=None,
                result=glm.GenerateContentResponse({'candidates': [{'content': {'parts': [{'text': 'world!'}], 'role': ''}, 'finish_reason': 0, 'safety_ratings': [], 'token_count': 0, 'grounding_attributions': []}]}),
            )"""
        )
        self.assertEqual(expected, result)

    def test_repr_for_streaming_start_to_finish(self):
        chunks = ["first", " second", " third"]
        self.responses["stream_generate_content"] = [(simple_response(text) for text in chunks)]

        model = generative_models.GenerativeModel("gemini-pro")
        response = model.generate_content("Hello", stream=True)
        iterator = iter(response)

        result1 = repr(response)
        expected1 = textwrap.dedent(
            """\
            response:
            GenerateContentResponse(
                done=False,
                iterator=<generator>,
                result=glm.GenerateContentResponse({'candidates': [{'content': {'parts': [{'text': 'first'}], 'role': ''}, 'finish_reason': 0, 'safety_ratings': [], 'token_count': 0, 'grounding_attributions': []}]}),
            )"""
        )
        self.assertEqual(expected1, result1)

        next(iterator)
        result2 = repr(response)
        expected2 = textwrap.dedent(
            """\
            response:
            GenerateContentResponse(
                done=False,
                iterator=<generator>,
                result=glm.GenerateContentResponse({'candidates': [{'content': {'parts': [{'text': 'first second'}], 'role': ''}, 'index': 0, 'citation_metadata': {'citation_sources': []}, 'finish_reason': 0, 'safety_ratings': [], 'token_count': 0, 'grounding_attributions': []}], 'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}}),
            )"""
        )
        self.assertEqual(expected2, result2)

        next(iterator)
        result3 = repr(response)
        expected3 = textwrap.dedent(
            """\
            response:
            GenerateContentResponse(
                done=False,
                iterator=<generator>,
                result=glm.GenerateContentResponse({'candidates': [{'content': {'parts': [{'text': 'first second third'}], 'role': ''}, 'index': 0, 'citation_metadata': {'citation_sources': []}, 'finish_reason': 0, 'safety_ratings': [], 'token_count': 0, 'grounding_attributions': []}], 'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}}),
            )"""
        )
        self.assertEqual(expected3, result3)

    def test_repr_error_info_for_stream_prompt_feedback_blocked(self):
        # response._error => BlockedPromptException
        chunks = [
            glm.GenerateContentResponse(
                {
                    "prompt_feedback": {"block_reason": "SAFETY"},
                }
            )
        ]
        self.responses["stream_generate_content"] = [(chunk for chunk in chunks)]

        model = generative_models.GenerativeModel("gemini-pro")
        response = model.generate_content("Bad stuff!", stream=True)

        result = repr(response)
        expected = textwrap.dedent(
            """\
            response:
            GenerateContentResponse(
                done=False,
                iterator=<generator>,
                result=glm.GenerateContentResponse({'prompt_feedback': {'block_reason': 1, 'safety_ratings': []}, 'candidates': []}),
            ),
            error=<BlockedPromptException> prompt_feedback {
              block_reason: SAFETY
            }
            """
        )
        self.assertEqual(expected, result)

    def test_repr_error_info_for_chat_error_in_stream(self):
        # response._error => ValueError
        def throw():
            for c in "123":
                yield simple_response(c)
            raise ValueError()

        def no_throw():
            for c in "abc":
                yield simple_response(c)

        self.responses["stream_generate_content"] = [
            no_throw(),
            throw(),
            no_throw(),
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()

        # Send a message, the response is okay..
        chat.send_message("hello1", stream=True).resolve()

        # Send a second message, it fails
        response = chat.send_message("hello2", stream=True)
        with self.assertRaises(ValueError):
            # Iteration fails
            for chunk in response:
                pass

        result = repr(response)
        expected = textwrap.dedent(
            """\
            response:
            GenerateContentResponse(
                done=True,
                iterator=None,
                result=glm.GenerateContentResponse({'candidates': [{'content': {'parts': [{'text': '123'}], 'role': ''}, 'index': 0, 'citation_metadata': {'citation_sources': []}, 'finish_reason': 0, 'safety_ratings': [], 'token_count': 0, 'grounding_attributions': []}], 'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}}),
            ),
            error=<ValueError> """
        )
        self.assertEqual(expected, result)

    def test_repr_error_info_for_chat_streaming_unexpected_stop(self):
        # response._error => StopCandidateException
        self.responses["stream_generate_content"] = [
            iter(
                [
                    simple_response("a"),
                    simple_response("b"),
                    simple_response("c"),
                    glm.GenerateContentResponse(
                        {
                            "candidates": [{"finish_reason": "SAFETY"}],
                        }
                    ),
                ]
            )
        ]

        model = generative_models.GenerativeModel("gemini-pro-vision")
        chat = model.start_chat()

        response = chat.send_message("hello", stream=True)
        for chunk in response:
            # The result doesn't know it's a chat result so it can't throw.
            # Unless we give it some way to know?
            pass

        with self.assertRaises(generation_types.BrokenResponseError):
            # But when preparing the next message, we can throw:
            response = chat.send_message("hello2", stream=True)

        result = repr(response)
        expected = textwrap.dedent(
            """\
            response:
            GenerateContentResponse(
                done=True,
                iterator=None,
                result=glm.GenerateContentResponse({'candidates': [{'content': {'parts': [{'text': 'abc'}], 'role': ''}, 'finish_reason': 3, 'index': 0, 'citation_metadata': {'citation_sources': []}, 'safety_ratings': [], 'token_count': 0, 'grounding_attributions': []}], 'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}}),
            ),
            error=<StopCandidateException> index: 0
            content {
              parts {
                text: "abc"
              }
            }
            finish_reason: SAFETY
            citation_metadata {
            }
            """
        )
        self.assertEqual(expected, result)

    def test_repr_for_multi_turn_chat(self):
        # Multi turn chat
        model = generative_models.GenerativeModel("gemini-pro")
        chat = model.start_chat()

        self.responses["generate_content"] = [
            simple_response("first"),
            simple_response("second"),
            simple_response("third"),
        ]

        msg1 = "I really like fantasy books."
        response = chat.send_message(msg1)

        msg2 = "I also like this image."
        response = chat.send_message([msg2, PIL.Image.open(TEST_IMAGE_PATH)])

        msg3 = "What things do I like?."
        response = chat.send_message(msg3)

        result = repr(chat)
        expected = textwrap.dedent(
            """\
            ChatSession(
                model=genai.GenerativeModel(
                    model_name='models/gemini-pro',
                    generation_config={},
                    safety_settings={},
                    tools=None,
                ),
                history=[glm.Content({'parts': [{'text': 'I really like fantasy books.'}], 'role': 'user'}), glm.Content({'parts': [{'text': 'first'}], 'role': 'model'}), glm.Content({'parts': [{'text': 'I also like this image.'}, {'inline_data': {'data': 'iVBORw0KGgoA...AAElFTkSuQmCC', 'mime_type': 'image/png'}}], 'role': 'user'}), glm.Content({'parts': [{'text': 'second'}], 'role': 'model'}), glm.Content({'parts': [{'text': 'What things do I like?.'}], 'role': 'user'}), glm.Content({'parts': [{'text': 'third'}], 'role': 'model'})]
            )"""
        )
        self.assertEqual(expected, result)

    def test_repr_for_incomplete_streaming_chat(self):
        # Multi turn chat
        model = generative_models.GenerativeModel("gemini-pro")
        chat = model.start_chat()

        self.responses["stream_generate_content"] = [
            (chunk for chunk in [simple_response("a"), simple_response("b"), simple_response("c")])
        ]

        msg1 = "I really like fantasy books."
        response = chat.send_message(msg1, stream=True)

        result = repr(chat)
        expected = textwrap.dedent(
            """\
            ChatSession(
                model=genai.GenerativeModel(
                    model_name='models/gemini-pro',
                    generation_config={},
                    safety_settings={},
                    tools=None,
                ),
                history=[glm.Content({'parts': [{'text': 'I really like fantasy books.'}], 'role': 'user'}), <STREAMING IN PROGRESS>]
            )"""
        )
        self.assertEqual(expected, result)

    def test_repr_for_broken_streaming_chat(self):
        # Multi turn chat
        model = generative_models.GenerativeModel("gemini-pro")
        chat = model.start_chat()

        self.responses["stream_generate_content"] = [
            (
                chunk
                for chunk in [
                    simple_response("first"),
                    # FinishReason.SAFETY = 3
                    glm.GenerateContentResponse(
                        {
                            "candidates": [
                                {"finish_reason": 3, "content": {"parts": [{"text": "second"}]}}
                            ]
                        }
                    ),
                ]
            )
        ]

        msg1 = "I really like fantasy books."
        response = chat.send_message(msg1, stream=True)

        for chunk in response:
            pass

        result = repr(chat)
        expected = textwrap.dedent(
            """\
            ChatSession(
                model=genai.GenerativeModel(
                    model_name='models/gemini-pro',
                    generation_config={},
                    safety_settings={},
                    tools=None,
                ),
                history=[glm.Content({'parts': [{'text': 'I really like fantasy books.'}], 'role': 'user'}), <STREAMING ERROR>]
            )"""
        )
        self.assertEqual(expected, result)

    def test_count_tokens_called_with_request_options(self):
        self.client.count_tokens = unittest.mock.MagicMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        self.responses["count_tokens"] = [glm.CountTokensResponse(total_tokens=7)]
        model = generative_models.GenerativeModel("gemini-pro-vision")
        model.count_tokens([{"role": "user", "parts": ["hello"]}], request_options=request_options)

        self.client.count_tokens.assert_called_once_with(request, **request_options)


if __name__ == "__main__":
    absltest.main()
