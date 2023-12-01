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
import datetime
import dataclasses
import pathlib
import pytz
from typing import Any, Union, Iterable
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import google.ai.generativelanguage as glm
from google.api_core import operation

from google.generativeai import models
from google.generativeai import client
from google.generativeai.types import model_types

import pandas as pd

HERE = pathlib.Path(__file__).parent


class UnitTests(parameterized.TestCase):
    def setUp(self):
        self.client = unittest.mock.MagicMock()

        client._client_manager.model_client = self.client

        # TODO(markdaoust): Check if typechecking works better if wee define this as a
        #                   subclass of `glm.ModelServiceClient`, would pyi files for `glm` help?
        def add_client_method(f):
            name = f.__name__
            setattr(self.client, name, f)
            return f

        self.observed_requests = []
        self.responses = {}

        @add_client_method
        def get_model(request: Union[glm.GetModelRequest, None] = None, *, name=None) -> glm.Model:
            if request is None:
                request = glm.GetModelRequest(name=name)
            self.assertIsInstance(request, glm.GetModelRequest)
            self.observed_requests.append(request)
            response = copy.copy(self.responses["get_model"])
            return response

        @add_client_method
        def get_tuned_model(
            request: Union[glm.GetTunedModelRequest, None] = None, *, name=None
        ) -> glm.TunedModel:
            if request is None:
                request = glm.GetTunedModelRequest(name=name)
            self.assertIsInstance(request, glm.GetTunedModelRequest)
            self.observed_requests.append(request)
            response = copy.copy(self.responses["get_tuned_model"])
            return response

        @add_client_method
        def list_models(
            request: Union[glm.ListModelsRequest, None] = None,
            *,
            page_size=None,
            page_token=None,
        ) -> glm.ListModelsResponse:
            if request is None:
                request = glm.ListModelsRequest(page_size=page_size, page_token=page_token)
            self.assertIsInstance(request, glm.ListModelsRequest)
            self.observed_requests.append(request)
            response = self.responses["list_models"]
            return (item for item in response)

        @add_client_method
        def list_tuned_models(
            request: glm.ListTunedModelsRequest = None,
            *,
            page_size=None,
            page_token=None,
        ) -> Iterable[glm.TunedModel]:
            if request is None:
                request = glm.ListTunedModelsRequest(page_size=page_size, page_token=page_token)
            self.assertIsInstance(request, glm.ListTunedModelsRequest)
            self.observed_requests.append(request)
            response = self.responses["list_tuned_models"]
            return (item for item in response)

        @add_client_method
        def update_tuned_model(request: glm.UpdateTunedModelRequest) -> glm.TunedModel:
            self.observed_requests.append(request)
            response = self.responses.get("update_tuned_model", None)
            if response is None:
                response = request.tuned_model
            return response

        @add_client_method
        def delete_tuned_model(name):
            request = glm.DeleteTunedModelRequest(name=name)
            self.observed_requests.append(request)
            response = True
            return response

        @add_client_method
        def create_tuned_model(request):
            request = glm.CreateTunedModelRequest(request)
            self.observed_requests.append(request)
            return self.responses["create_tuned_model"]

    def test_decode_tuned_model_time_round_trip(self):
        example_dt = datetime.datetime(2000, 1, 2, 3, 4, 5, 600_000, pytz.UTC)
        tuned_model = glm.TunedModel(name="tunedModels/house-mouse-001", create_time=example_dt)
        tuned_model = model_types.decode_tuned_model(tuned_model)
        self.assertEqual(tuned_model.create_time, example_dt)

    @parameterized.named_parameters(
        ["simple", "models/fake-bison-001"],
        ["simple-tuned", "tunedModels/my-pig-001"],
        ["model-instance", glm.Model(name="models/fake-bison-001")],
        ["tuned-model-instance", glm.TunedModel(name="tunedModels/my-pig-001")],
    )
    def test_get_model(self, name):
        self.responses = {
            "get_model": glm.Model(name="models/fake-bison-001"),
            "get_tuned_model": glm.TunedModel(name="tunedModels/my-pig-001"),
        }

        model = models.get_model(name)
        if self.observed_requests[0].name.startswith("models/"):
            self.assertIsInstance(model, model_types.Model)
        else:
            self.assertIsInstance(model, model_types.TunedModel)

    @parameterized.named_parameters(
        ["simple", "mystery-bison-001"],
        ["model-instance", glm.Model(name="how?-bison-001")],
    )
    def test_fail_with_unscoped_model_name(self, name):
        with self.assertRaises(ValueError):
            model = models.get_model(name)

    def test_list_models(self):
        # The low level lib wraps the response in an iterable, so this is a fair test.
        self.responses = {
            "list_models": [
                glm.Model(name="models/fake-bison-001"),
                glm.Model(name="models/fake-bison-002"),
                glm.Model(name="models/fake-bison-003"),
            ]
        }

        found_models = list(models.list_models())
        self.assertLen(found_models, 3)
        for m in found_models:
            self.assertIsInstance(m, model_types.Model)

    def test_list_tuned_models(self):
        self.responses = {
            # The low level lib wraps the response in an iterable, so this is a fair test.
            "list_tuned_models": [
                glm.TunedModel(name="tunedModels/my-pig-001"),
                glm.TunedModel(name="tunedModels/my-pig-002"),
                glm.TunedModel(name="tunedModels/my-pig-003"),
            ]
        }
        found_models = list(models.list_tuned_models())
        self.assertLen(found_models, 3)
        for m in found_models:
            self.assertIsInstance(m, model_types.TunedModel)

    @parameterized.named_parameters(
        [
            "edited-glm-model",
            glm.TunedModel(
                name="tunedModels/my-pig-001",
                description="Trained on my data",
            ),
            None,
        ],
        [
            "name-and-dict",
            "tunedModels/my-pig-001",
            {"description": "Trained on my data"},
        ],
    )
    def test_update_tuned_model_basics(self, tuned_model, updates):
        self.responses["get_tuned_model"] = glm.TunedModel(name="tunedModels/my-pig-001")
        # No self.responses['update_tuned_model'] the mock just returns the input.
        updated_model = models.update_tuned_model(tuned_model, updates)
        updated_model.description = "Trained on my data"

    @parameterized.named_parameters(
        [
            "dict",
            {"tuning_task": {"hyperparameters": {"batch_size": 8}}},
        ],
        [
            "flat-dict",
            {"tuning_task.hyperparameters.batch_size": 8},
        ],
    )
    def test_update_tuned_model_nested_fields(self, updates):
        self.responses["get_tuned_model"] = glm.TunedModel(
            name="tunedModels/my-pig-001", base_model="models/dance-monkey-007"
        )

        result = models.update_tuned_model("tunedModels/my-pig-001", updates)
        self.assertEqual(
            result,
            model_types.TunedModel(
                name="tunedModels/my-pig-001",
                source_model="models/dance-monkey-007",
                base_model="models/dance-monkey-007",
                tuning_task=model_types.TuningTask(
                    hyperparameters=model_types.Hyperparameters(
                        batch_size=8, learning_rate=0, epoch_count=0
                    ),
                    snapshots=[],
                ),
            ),
        )

    @parameterized.named_parameters(
        ["name", "tunedModels/bipedal-pangolin-223"],
        [
            "glm.TunedModel",
            glm.TunedModel(name="tunedModels/bipedal-pangolin-223"),
        ],
        [
            "models.TunedModel",
            model_types.TunedModel(name="tunedModels/bipedal-pangolin-223"),
        ],
    )
    def test_delete_tuned_model(self, model):
        models.delete_tuned_model(model)
        self.assertEqual(self.observed_requests[0].name, "tunedModels/bipedal-pangolin-223")

    @parameterized.named_parameters(
        ["simple", "2000-01-01T01:01:01.123456Z", 123456],
        ["zeros-right", "2000-01-01T01:01:01.100000Z", 100000],
        ["zeros-left", "2000-01-01T01:01:01.000001Z", 1],
        ["short", "2000-01-01T01:01:01.12Z", 120000],
        ["long", "2000-01-01T01:01:01.1234567899999999Z", 123457],
    )
    def test_decode_micros(self, time_str, micros):
        time = {"time": time_str}
        model_types.idecode_time(time, "time")
        self.assertEqual(time["time"].microsecond, micros)

    def test_decode_tuned_model(self):
        out_fields = glm.TunedModel(
            state=glm.TunedModel.State.CREATING,
            create_time="2000-01-01T01:01:01.0Z",
            update_time="2001-01-01T01:01:01.0Z",
            tuning_task=glm.TuningTask(
                hyperparameters=glm.Hyperparameters(
                    batch_size=72, epoch_count=1, learning_rate=0.1
                ),
                start_time="2002-01-01T01:01:01.0Z",
                complete_time="2003-01-01T01:01:01.0Z",
                snapshots=[
                    glm.TuningSnapshot(
                        step=1,
                        epoch=1,
                        compute_time="2004-01-01T01:01:01.0Z",
                    ),
                    glm.TuningSnapshot(
                        step=2,
                        epoch=1,
                        compute_time="2005-01-01T01:01:01.0Z",
                    ),
                ],
            ),
        )

        decoded = model_types.decode_tuned_model(out_fields)
        self.assertEqual(decoded.state, glm.TunedModel.State.CREATING)
        self.assertEqual(decoded.create_time.year, 2000)
        self.assertEqual(decoded.update_time.year, 2001)
        self.assertIsInstance(decoded.tuning_task.hyperparameters, model_types.Hyperparameters)
        self.assertEqual(decoded.tuning_task.hyperparameters.batch_size, 72)
        self.assertIsInstance(decoded.tuning_task, model_types.TuningTask)
        self.assertEqual(decoded.tuning_task.start_time.year, 2002)
        self.assertEqual(decoded.tuning_task.complete_time.year, 2003)
        self.assertIsInstance(decoded.tuning_task.snapshots, list)
        self.assertEqual(decoded.tuning_task.snapshots[0]["compute_time"].year, 2004)
        self.assertEqual(decoded.tuning_task.snapshots[1]["compute_time"].year, 2005)

    @parameterized.named_parameters(
        ["simple", glm.TunedModel(base_model="models/swim-fish-000")],
        [
            "nested",
            glm.TunedModel(
                tuned_model_source={
                    "tuned_model": "tunedModels/hidden-fish-55",
                    "base_model": "models/swim-fish-000",
                }
            ),
        ],
    )
    def test_smoke_decode_tuned_model(self, model):
        decoded = model_types.decode_tuned_model(model)
        self.assertEqual(decoded.base_model, "models/swim-fish-000")
        self.assertFalse(decoded.source_model is None)

    def test_smoke_create_tuned_model(self):
        self.responses["create_tuned_model"] = operation.Operation(
            operation.operations_pb2.Operation(), None, None, None
        )
        models.create_tuned_model(
            source_model="models/sneaky-fox-001",
            temperature=0.5,
            batch_size=32,
            training_data=[
                ("in", "out"),
                {"text_input": "in", "output": "out"},
                glm.TuningExample(text_input="in", output="out"),
            ],
        )
        req = self.observed_requests[-1]
        self.assertEqual(req.tuned_model.base_model, "models/sneaky-fox-001")
        self.assertEqual(self.observed_requests[-1].tuned_model.temperature, 0.5)
        self.assertEqual(req.tuned_model.tuning_task.hyperparameters.batch_size, 32)
        self.assertLen(req.tuned_model.tuning_task.training_data.examples.examples, 3)

    @parameterized.named_parameters(
        ["simple", glm.TunedModel(base_model="models/swim-fish-000")],
        [
            "nested",
            glm.TunedModel(
                tuned_model_source={
                    "tuned_model": "tunedModels/hidden-fish-55",
                    "base_model": "models/swim-fish-000",
                }
            ),
        ],
    )
    def test_create_tuned_model_on_tuned_model(self, tuned_source):
        self.responses["create_tuned_model"] = operation.Operation(
            operation.operations_pb2.Operation(), None, None, None
        )
        self.responses["get_tuned_model"] = tuned_source
        models.create_tuned_model(source_model="tunedModels/swim-fish-001", training_data=[])

        self.assertEqual(
            self.observed_requests[-1].tuned_model.tuned_model_source.tuned_model,
            "tunedModels/swim-fish-001",
        )
        self.assertEqual(
            self.observed_requests[-1].tuned_model.tuned_model_source.base_model,
            "models/swim-fish-000",
        )

    @parameterized.named_parameters(
        [
            "glm",
            glm.Dataset(
                examples=glm.TuningExamples(
                    examples=[
                        {"text_input": "a", "output": "1"},
                        {"text_input": "b", "output": "2"},
                        {"text_input": "c", "output": "3"},
                    ]
                )
            ),
        ],
        [
            "list",
            [
                ("a", "1"),
                {"text_input": "b", "output": "2"},
                glm.TuningExample({"text_input": "c", "output": "3"}),
            ],
        ],
        ["dict", {"text_input": ["a", "b", "c"], "output": ["1", "2", "3"]}],
        [
            "dict_custom_keys",
            {"my_inputs": ["a", "b", "c"], "my_outputs": ["1", "2", "3"]},
            "my_inputs",
            "my_outputs",
        ],
        [
            "pd.DataFrame",
            pd.DataFrame(
                [
                    {"text_input": "a", "output": "1"},
                    {"text_input": "b", "output": "2"},
                    {"text_input": "c", "output": "3"},
                ]
            ),
        ],
        ["csv-path-string", str(HERE / "test.csv")],
        ["csv-path", HERE / "test.csv"],
        ["json-file-1", HERE / "test1.json"],
        ["json-file-2", HERE / "test2.json"],
        ["json-file-3", HERE / "test3.json"],
        [
            "json-url",
            "https://storage.googleapis.com/generativeai-downloads/data/test1.json",
        ],
        [
            "csv-url",
            "https://storage.googleapis.com/generativeai-downloads/data/test.csv",
        ],
        [
            "sheet-share",
            "https://docs.google.com/spreadsheets/d/1OffcVSqN6X-RYdWLGccDF3KtnKoIpS7O_9cZbicKK4A/edit?usp=sharing",
        ],
        [
            "sheet-export-csv",
            "https://docs.google.com/spreadsheets/d/1OffcVSqN6X-RYdWLGccDF3KtnKoIpS7O_9cZbicKK4A/export?format=csv",
        ],
        [
            "sheet-with-tab",
            "https://docs.google.com/spreadsheets/d/118LXTS3RIkS4yAO68c-cMPP4PwLFTxKYj4R43R7dU0E/edit#gid=1526779134",
        ],
    )
    def test_create_dataset(self, data, ik="text_input", ok="output"):
        ds = model_types.encode_tuning_data(data, input_key=ik, output_key=ok)

        expect = glm.Dataset(
            examples=glm.TuningExamples(
                examples=[
                    {"text_input": "a", "output": "1"},
                    {"text_input": "b", "output": "2"},
                    {"text_input": "c", "output": "3"},
                ]
            )
        )
        self.assertEqual(expect, ds)


if __name__ == "__main__":
    absltest.main()
