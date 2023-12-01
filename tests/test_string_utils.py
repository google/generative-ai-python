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

import dataclasses
import pprint
import textwrap
from typing import Any, List
import unittest


from google.generativeai import string_utils

from absl.testing import parameterized


@string_utils.prettyprint
@dataclasses.dataclass
class MyClass:
    a: int
    b: float
    c: List[int]
    d: Any


class OperationsTests(parameterized.TestCase):
    def test_simple(self):
        m = MyClass(a=1, b=1 / 3, c=[0, 1, 2, 3, 4, 5], d={"a": 1, "b": 2})

        result = str(m)
        expected = textwrap.dedent(
            """
            MyClass(a=1,
                    b=0.3333333333333333,
                    c=[0, 1, 2, 3, 4, 5],
                    d={'a': 1, 'b': 2})"""
        )[1:]
        self.assertEqual(expected, result)
        self.assertEqual(pprint.pformat(m), result)
        self.assertEqual(repr(m), result)

    def test_long_list(self):
        m = MyClass(a=1, b=1 / 3, c=[1, 2, 3] * 10, d={"a": 1, "b": 2})
        expected = textwrap.dedent(
            """
            MyClass(a=1,
                    b=0.3333333333333333,
                    c=[...],
                    d={'a': 1, 'b': 2})"""
        )[1:]
        self.assertEqual(expected, str(m))

    def test_nested(self):
        m1 = MyClass(a=1, b=1 / 3, c=[1, 2, 3], d={"a": 1, "b": 2})
        m2 = MyClass(a=1, b=1 / 3, c=[1, 2, 3], d=m1)

        expected = textwrap.dedent(
            """
            MyClass(a=1,
                    b=0.3333333333333333,
                    c=[1, 2, 3],
                    d=MyClass(a=1,
                              b=0.3333333333333333,
                              c=[1, 2, 3],
                              d={'a': 1, 'b': 2}))"""
        )[1:]
        result = str(m2)
        self.assertEqual(expected, result)
        self.assertEqual(pprint.pformat(m2), result)
        self.assertEqual(repr(m2), result)

    def test_long_obj(self):
        m = MyClass(a=1, b=1 / 3, c=[1, 2, 3], d=None)
        m = MyClass(a=1, b=1 / 3, c=[1, 2, 3], d=m)
        m = MyClass(a=1, b=1 / 3, c=[1, 2, 3], d=m)
        m = MyClass(a=1, b=1 / 3, c=[1, 2, 3], d=m)
        m = MyClass(a=1, b=1 / 3, c=[1, 2, 3], d=m)

        expected = textwrap.dedent(
            """
            MyClass(a=1,
                    b=0.3333333333333333,
                    c=[1, 2, 3],
                    d=MyClass(...))"""
        )[1:]
        self.assertEqual(expected, str(m))

    def test_recursive(self):
        m = MyClass(a=1, b=1 / 3, c=[1, 2, 3], d=None)
        m.d = m

        expected = textwrap.dedent(
            """
            MyClass(a=1,
                    b=0.3333333333333333,
                    c=[1, 2, 3],
                    d=...)"""
        )[1:]
        result = str(m)
        self.assertEqual(expected, result)
        self.assertEqual(pprint.pformat(m), result)
        self.assertEqual(repr(m), result)
