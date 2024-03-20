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
import pathlib
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import google.ai.generativelanguage as glm
from google.generativeai.types import content_types
import IPython.display
import PIL.Image


HERE = pathlib.Path(__file__).parent
TEST_PNG_PATH = HERE / "test_img.png"
TEST_PNG_URL = "https://storage.googleapis.com/generativeai-downloads/data/test_img.png"
TEST_PNG_DATA = TEST_PNG_PATH.read_bytes()

TEST_JPG_PATH = HERE / "test_img.jpg"
TEST_JPG_URL = "https://storage.googleapis.com/generativeai-downloads/data/test_img.jpg"
TEST_JPG_DATA = TEST_JPG_PATH.read_bytes()


# simple test function
def datetime():
    "Returns the current UTC date and time."


class UnitTests(parameterized.TestCase):
    @parameterized.named_parameters(
        ["PIL", PIL.Image.open(TEST_PNG_PATH)],
        ["IPython", IPython.display.Image(filename=TEST_PNG_PATH)],
    )
    def test_png_to_blob(self, image):
        blob = content_types.image_to_blob(image)
        self.assertIsInstance(blob, glm.Blob)
        self.assertEqual(blob.mime_type, "image/png")
        self.assertStartsWith(blob.data, b"\x89PNG")

    @parameterized.named_parameters(
        ["PIL", PIL.Image.open(TEST_JPG_PATH)],
        ["IPython", IPython.display.Image(filename=TEST_JPG_PATH)],
    )
    def test_jpg_to_blob(self, image):
        blob = content_types.image_to_blob(image)
        self.assertIsInstance(blob, glm.Blob)
        self.assertEqual(blob.mime_type, "image/jpeg")
        self.assertStartsWith(blob.data, b"\xff\xd8\xff\xe0\x00\x10JFIF")

    @parameterized.named_parameters(
        ["BlobDict", {"mime_type": "image/png", "data": TEST_PNG_DATA}],
        ["glm.Blob", glm.Blob(mime_type="image/png", data=TEST_PNG_DATA)],
        ["Image", IPython.display.Image(filename=TEST_PNG_PATH)],
    )
    def test_to_blob(self, example):
        blob = content_types.to_blob(example)
        self.assertIsInstance(blob, glm.Blob)
        self.assertEqual(blob.mime_type, "image/png")
        self.assertStartsWith(blob.data, b"\x89PNG")

    @parameterized.named_parameters(
        ["dict", {"text": "Hello world!"}],
        ["glm.Part", glm.Part(text="Hello world!")],
        ["str", "Hello world!"],
    )
    def test_to_part(self, example):
        part = content_types.to_part(example)
        self.assertIsInstance(part, glm.Part)
        self.assertEqual(part.text, "Hello world!")

    @parameterized.named_parameters(
        ["Image", IPython.display.Image(filename=TEST_PNG_PATH)],
        ["BlobDict", {"mime_type": "image/png", "data": TEST_PNG_DATA}],
        [
            "PartDict",
            {"inline_data": {"mime_type": "image/png", "data": TEST_PNG_DATA}},
        ],
    )
    def test_img_to_part(self, example):
        blob = content_types.to_part(example).inline_data
        self.assertIsInstance(blob, glm.Blob)
        self.assertEqual(blob.mime_type, "image/png")
        self.assertStartsWith(blob.data, b"\x89PNG")

    @parameterized.named_parameters(
        ["glm.Content", glm.Content(parts=[{"text": "Hello world!"}])],
        ["ContentDict", {"parts": [{"text": "Hello world!"}]}],
        ["ContentDict-str", {"parts": ["Hello world!"]}],
        ["list[parts]", [{"text": "Hello world!"}]],
        ["list[str]", ["Hello world!"]],
        ["iterator[parts]", iter([{"text": "Hello world!"}])],
        ["part", {"text": "Hello world!"}],
        ["str", "Hello world!"],
    )
    def test_to_content(self, example):
        content = content_types.to_content(example)
        part = content.parts[0]

        self.assertLen(content.parts, 1)
        self.assertIsInstance(part, glm.Part)
        self.assertEqual(part.text, "Hello world!")

    @parameterized.named_parameters(
        ["ContentDict", {"parts": [PIL.Image.open(TEST_PNG_PATH)]}],
        ["list[Image]", [PIL.Image.open(TEST_PNG_PATH)]],
        ["Image", PIL.Image.open(TEST_PNG_PATH)],
    )
    def test_img_to_content(self, example):
        content = content_types.to_content(example)
        blob = content.parts[0].inline_data
        self.assertLen(content.parts, 1)
        self.assertIsInstance(blob, glm.Blob)
        self.assertEqual(blob.mime_type, "image/png")
        self.assertStartsWith(blob.data, b"\x89PNG")

    @parameterized.named_parameters(
        ["glm.Content", glm.Content(parts=[{"text": "Hello world!"}])],
        ["ContentDict", {"parts": [{"text": "Hello world!"}]}],
        ["ContentDict-str", {"parts": ["Hello world!"]}],
    )
    def test_strict_to_content(self, example):
        content = content_types.strict_to_content(example)
        part = content.parts[0]

        self.assertLen(content.parts, 1)
        self.assertIsInstance(part, glm.Part)
        self.assertEqual(part.text, "Hello world!")

    @parameterized.named_parameters(
        ["list[parts]", [{"text": "Hello world!"}]],
        ["list[str]", ["Hello world!"]],
        ["iterator[parts]", iter([{"text": "Hello world!"}])],
        ["part", {"text": "Hello world!"}],
        ["str", "Hello world!"],
    )
    def test_strict_to_contents_fails(self, examples):
        with self.assertRaises(TypeError):
            content_types.strict_to_content(examples)

    @parameterized.named_parameters(
        ["glm.Content", [glm.Content(parts=[{"text": "Hello world!"}])]],
        ["ContentDict", [{"parts": [{"text": "Hello world!"}]}]],
        ["ContentDict-unwraped", [{"parts": ["Hello world!"]}]],
        ["ContentDict+str-part", [{"parts": "Hello world!"}]],
    )
    def test_to_contents(self, example):
        contents = content_types.to_contents(example)
        part = contents[0].parts[0]

        self.assertLen(contents, 1)

        self.assertLen(contents[0].parts, 1)
        self.assertIsInstance(part, glm.Part)
        self.assertEqual(part.text, "Hello world!")

    def test_dict_to_content_fails(self):
        with self.assertRaises(KeyError):
            content_types.to_content({"bad": "dict"})

    @parameterized.named_parameters(
        [
            "ContentDict",
            [{"parts": [{"inline_data": PIL.Image.open(TEST_PNG_PATH)}]}],
        ],
        ["ContentDict-unwraped", [{"parts": [PIL.Image.open(TEST_PNG_PATH)]}]],
        ["Image", PIL.Image.open(TEST_PNG_PATH)],
    )
    def test_img_to_contents(self, example):
        contents = content_types.to_contents(example)
        blob = contents[0].parts[0].inline_data

        self.assertLen(contents, 1)
        self.assertLen(contents[0].parts, 1)
        self.assertIsInstance(blob, glm.Blob)
        self.assertEqual(blob.mime_type, "image/png")
        self.assertStartsWith(blob.data, b"\x89PNG")

    @parameterized.named_parameters(
        [
            "FunctionLibrary",
            content_types.FunctionLibrary(
                tools=glm.Tool(
                    function_declarations=[
                        glm.FunctionDeclaration(
                            name="datetime", description="Returns the current UTC date and time."
                        )
                    ]
                )
            ),
        ],
        [
            "IterableTool-Tool",
            [
                content_types.Tool(
                    function_declarations=[
                        glm.FunctionDeclaration(
                            name="datetime", description="Returns the current UTC date and time."
                        )
                    ]
                )
            ],
        ],
        [
            "IterableTool-glm.Tool",
            [
                glm.Tool(
                    function_declarations=[
                        glm.FunctionDeclaration(
                            name="datetime",
                            description="Returns the current UTC date and time.",
                        )
                    ]
                )
            ],
        ],
        [
            "IterableTool-ToolDict",
            [
                dict(
                    function_declarations=[
                        dict(
                            name="datetime",
                            description="Returns the current UTC date and time.",
                        )
                    ]
                )
            ],
        ],
        [
            "IterableTool-IterableFD",
            [
                [
                    glm.FunctionDeclaration(
                        name="datetime",
                        description="Returns the current UTC date and time.",
                    )
                ]
            ],
        ],
        [
            "IterableTool-FD",
            [
                glm.FunctionDeclaration(
                    name="datetime",
                    description="Returns the current UTC date and time.",
                )
            ],
        ],
        [
            "Tool",
            content_types.Tool(
                function_declarations=[
                    glm.FunctionDeclaration(
                        name="datetime", description="Returns the current UTC date and time."
                    )
                ]
            ),
        ],
        [
            "glm.Tool",
            glm.Tool(
                function_declarations=[
                    glm.FunctionDeclaration(
                        name="datetime", description="Returns the current UTC date and time."
                    )
                ]
            ),
        ],
        [
            "ToolDict",
            dict(
                function_declarations=[
                    dict(name="datetime", description="Returns the current UTC date and time.")
                ]
            ),
        ],
        [
            "IterableFD-FD",
            [
                content_types.FunctionDeclaration(
                    name="datetime", description="Returns the current UTC date and time."
                )
            ],
        ],
        [
            "IterableFD-CFD",
            [
                content_types.CallableFunctionDeclaration(
                    name="datetime",
                    description="Returns the current UTC date and time.",
                    function=datetime,
                )
            ],
        ],
        [
            "IterableFD-dict",
            [dict(name="datetime", description="Returns the current UTC date and time.")],
        ],
        ["IterableFD-Callable", [datetime]],
        [
            "FD",
            content_types.FunctionDeclaration(
                name="datetime", description="Returns the current UTC date and time."
            ),
        ],
        [
            "CFD",
            content_types.CallableFunctionDeclaration(
                name="datetime",
                description="Returns the current UTC date and time.",
                function=datetime,
            ),
        ],
        [
            "glm.FD",
            glm.FunctionDeclaration(
                name="datetime", description="Returns the current UTC date and time."
            ),
        ],
        ["dict", dict(name="datetime", description="Returns the current UTC date and time.")],
        ["Callable", datetime],
    )
    def test_to_tools(self, tools):
        function_library = content_types.to_function_library(tools)
        if function_library is None:
            raise ValueError("This shouldn't happen")
        tools = function_library.to_proto()

        tools = type(tools[0]).to_dict(tools[0])
        tools["function_declarations"][0].pop("parameters", None)

        expected = dict(
            function_declarations=[
                dict(name="datetime", description="Returns the current UTC date and time.")
            ]
        )

        self.assertEqual(tools, expected)

    def test_two_fun_is_one_tool(self):
        def a():
            pass

        def b():
            pass

        function_library = content_types.to_function_library([a, b])
        if function_library is None:
            raise ValueError("This shouldn't happen")
        tools = function_library.to_proto()

        self.assertLen(tools, 1)
        self.assertLen(tools[0].function_declarations, 2)

    @parameterized.named_parameters(
        ["int", int, glm.Schema(type=glm.Type.INTEGER)],
        ["float", float, glm.Schema(type=glm.Type.NUMBER)],
        ["str", str, glm.Schema(type=glm.Type.STRING)],
        [
            "list",
            list[str],
            glm.Schema(
                type=glm.Type.ARRAY,
                items=glm.Schema(type=glm.Type.STRING),
            ),
        ],
        [
            "list-list-int",
            list[list[int]],
            glm.Schema(
                type=glm.Type.ARRAY,
                items=glm.Schema(
                    glm.Schema(
                        type=glm.Type.ARRAY,
                        items=glm.Schema(type=glm.Type.INTEGER),
                    ),
                ),
            ),
        ],
        ["dict", dict, glm.Schema(type=glm.Type.OBJECT)],
        ["dict-str-any", dict[str, Any], glm.Schema(type=glm.Type.OBJECT)],
    )
    def test_auto_schema(self, annotation, expected):
        def fun(a: annotation):
            pass

        cfd = content_types.FunctionDeclaration.from_function(fun)
        got = cfd.parameters.properties["a"]
        self.assertEqual(got, expected)


if __name__ == "__main__":
    absltest.main()
