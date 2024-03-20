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
#
# pytype: skip-file
r"""Api reference docs generation script, using tensorflow_docs

This script generates API reference docs for the reference doc generator.

$> pip install -U git+https://github.com/tensorflow/docs
$> python build_docs.py
"""

import os
import pathlib
import re
import textwrap
import typing


from absl import app
from absl import flags

import google
from google.ai import generativelanguage as glm

import grpc

# For showing the conditional imports and types in `content_types.py`
# grpc must be imported first.
typing.TYPE_CHECKING = True
from google import generativeai as palm


from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import yaml

glm.__doc__ = """\
This package, `google.ai.generativelanguage`, is a low-level auto-generated client library for the PaLM API.

```posix-terminal
pip install google.ai.generativelanguage
```

It is built using the same tooling as Google Cloud client libraries, and will be quite familiar if you've used
those before.

While we encourage Python users to access the PaLM API using the `google.generativeai` package (aka `palm`),
this lower level package is also available.

Each method in the PaLM API is connected to one of the client classes. Pass your API-key to the class' `client_options`
when initializing a client:

```
from google.ai import generativelanguage as glm

client = glm.DiscussServiceClient(
    client_options={'api_key':'YOUR_API_KEY'})
```

To call the api, pass an appropriate request-proto-object. For the `DiscussServiceClient.generate_message` pass
a `generativelanguage.GenerateMessageRequest` instance:

```
request = glm.GenerateMessageRequest(
    model='models/chat-bison-001',
    prompt=glm.MessagePrompt(
        messages=[glm.Message(content='Hello!')]))

client.generate_message(request)
```
```
candidates {
  author: "1"
  content: "Hello! How can I help you today?"
}
...
```

For simplicity:

* The API methods also accept key-word arguments.
* Anywhere you might pass a proto-object, the library will also accept simple python structures.

So the following is equivalent to the previous example:

```
client.generate_message(
    model='models/chat-bison-001',
    prompt={'messages':[{'content':'Hello!'}]})
```
```
candidates {
  author: "1"
  content: "Hello! How can I help you today?"
}
...
```
"""

HERE = pathlib.Path(__file__).parent

PROJECT_SHORT_NAME = "genai"
PROJECT_FULL_NAME = "Generative AI - Python"

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    default=str(HERE / "api/"),
    help="Where to write the resulting docs to.",
)

_SEARCH_HINTS = flags.DEFINE_bool(
    "search_hints", True, "Include metadata search hints in the generated files"
)

_SITE_PATH = flags.DEFINE_string("site_path", "/api/python", "Path prefix in the _toc.yaml")

_CODE_URL_PREFIX = flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/google/generative-ai-python/blob/master/google/generativeai",
    "where to find the project code",
)


class MyFilter:
    def __init__(self, base_dirs):
        self.filter_base_dirs = public_api.FilterBaseDirs(base_dirs)

    def drop_staticmethods(self, parent, children):
        parent = dict(parent.__dict__)
        for name, value in children:
            if not isinstance(parent.get(name, None), staticmethod):
                yield name, value

    def __call__(self, path, parent, children):
        if any("generativelanguage" in part for part in path) or "generativeai" in path:
            children = self.filter_base_dirs(path, parent, children)
            children = public_api.explicit_package_contents_filter(path, parent, children)

        if any("generativelanguage" in part for part in path):
            if "ServiceClient" in path[-1] or "ServiceAsyncClient" in path[-1]:
                children = list(self.drop_staticmethods(parent, children))

        return children


class MyDocGenerator(generate_lib.DocGenerator):
    def make_default_filters(self):
        return [
            # filter the api.
            public_api.FailIfNestedTooDeep(10),
            public_api.filter_module_all,
            public_api.add_proto_fields,
            public_api.filter_builtin_modules,
            public_api.filter_private_symbols,
            MyFilter(self._base_dir),  # Replaces: public_api.FilterBaseDirs(self._base_dir),
            public_api.FilterPrivateMap(self._private_map),
            public_api.filter_doc_controls_skip,
            public_api.ignore_typing,
        ]


def gen_api_docs():
    """Generates api docs for the generative-ai package."""
    for name in dir(google):
        if name not in ("generativeai", "ai"):
            delattr(google, name)
    google.__name__ = "google"
    google.__doc__ = textwrap.dedent(
        """\
        This is the top-level google namespace.
        """
    )

    doc_generator = MyDocGenerator(
        root_title=PROJECT_FULL_NAME,
        py_modules=[("google", google)],
        base_dir=(
            pathlib.Path(palm.__file__).parent,
            pathlib.Path(glm.__file__).parent.parent,
        ),
        code_url_prefix=(
            _CODE_URL_PREFIX.value,
            "https://github.com/googleapis/google-cloud-python/tree/main/packages/google-ai-generativelanguage/google/ai",
        ),
        search_hints=_SEARCH_HINTS.value,
        site_path=_SITE_PATH.value,
        callbacks=[],
    )

    out_path = pathlib.Path(_OUTPUT_DIR.value)
    doc_generator.build(out_path)

    # Fixup the toc file.
    toc_path = out_path / "google/_toc.yaml"
    toc = yaml.safe_load(toc_path.read_text())
    assert toc["toc"][0]["title"] == "google"
    toc["toc"] = toc["toc"][1:]
    toc["toc"][0]["title"] = "google.ai.generativelanguage"
    toc["toc"][0]["section"] = toc["toc"][0]["section"][1]["section"]
    toc["toc"][0], toc["toc"][1] = toc["toc"][1], toc["toc"][0]
    toc_path.write_text(yaml.dump(toc))

    # remove some dummy files and redirect them to `api/`
    (out_path / "google.md").unlink()
    (out_path / "google/ai.md").unlink()
    redirects_path = out_path / "_redirects.yaml"
    redirects = {"redirects": []}
    redirects["redirects"].insert(0, {"from": "/api/python/google/ai", "to": "/api/"})
    redirects["redirects"].insert(0, {"from": "/api/python/google", "to": "/api/"})
    redirects["redirects"].insert(0, {"from": "/api/python", "to": "/api/"})
    redirects_path.write_text(yaml.dump(redirects))

    # clear `oneof` junk from proto pages
    for fpath in out_path.rglob("*.md"):
        old_content = fpath.read_text()
        new_content = old_content
        new_content = re.sub(r"\.\. _oneof:.*?\n", "", new_content)
        new_content = re.sub(r".*?`oneof`_ ``_.*?\n", "", new_content, re.MULTILINE)
        new_content = re.sub(r"\.\. code-block:: python.*?\n", "", new_content)

        new_content = re.sub(r"generativelanguage_\w+\.types", "generativelanguage", new_content)

        if new_content != old_content:
            fpath.write_text(new_content)

    print("Output docs to: ", _OUTPUT_DIR.value)


def main(_):
    gen_api_docs()


if __name__ == "__main__":
    app.run(main)
