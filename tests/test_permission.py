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
from typing import Any, Optional, Union
import unittest
import unittest.mock as mock

import google.ai.generativelanguage as glm

from google.generativeai import retriever
from google.generativeai import permission
from google.generativeai import models
from google.generativeai.types import permission_types as permission_services
from google.generativeai.types import retriever_types as retriever_services
from google.generativeai.types import model_types as model_services

from google.generativeai import client
from absl.testing import absltest
from absl.testing import parameterized


class UnitTests(parameterized.TestCase):
    def setUp(self):
        self.client = unittest.mock.MagicMock()

        client._client_manager.clients["retriever"] = self.client
        client._client_manager.clients["permission"] = self.client
        client._client_manager.clients["model"] = self.client

        self.observed_requests = []

        self.responses = {}

        def add_client_method(f):
            name = f.__name__
            setattr(self.client, name, f)
            return f

        @add_client_method
        def create_corpus(
            request: glm.CreateCorpusRequest,
            **kwargs,
        ) -> glm.Corpus:
            self.observed_requests.append(request)
            return glm.Corpus(
                name="corpora/demo-corpus",
                display_name="demo-corpus",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def get_tuned_model(
            request: Optional[glm.GetTunedModelRequest] = None,
            *,
            name=None,
            **kwargs,
        ) -> glm.TunedModel:
            if request is None:
                request = glm.GetTunedModelRequest(name=name)
            self.assertIsInstance(request, glm.GetTunedModelRequest)
            self.observed_requests.append(request)
            response = copy.copy(self.responses["get_tuned_model"])
            return response

        @add_client_method
        def create_permission(
            request: glm.CreatePermissionRequest,
        ) -> glm.Permission:
            self.observed_requests.append(request)
            return glm.Permission(
                name="corpora/demo-corpus/permissions/123456789",
                role=permission_services.to_role("writer"),
                grantee_type=permission_services.to_grantee_type("everyone"),
            )

        @add_client_method
        def delete_permission(
            request: glm.DeletePermissionRequest,
        ) -> None:
            self.observed_requests.append(request)
            return None

        @add_client_method
        def get_permission(
            request: glm.GetPermissionRequest,
        ) -> glm.Permission:
            self.observed_requests.append(request)
            return glm.Permission(
                name="corpora/demo-corpus/permissions/123456789",
                role=permission_services.to_role("writer"),
                grantee_type=permission_services.to_grantee_type("everyone"),
            )

        @add_client_method
        def list_permissions(
            request: glm.ListPermissionsRequest,
        ) -> glm.ListPermissionsResponse:
            self.observed_requests.append(request)
            return [
                glm.Permission(
                    name="corpora/demo-corpus/permissions/123456789",
                    role=permission_services.to_role("writer"),
                    grantee_type=permission_services.to_grantee_type("everyone"),
                ),
                glm.Permission(
                    name="corpora/demo-corpus/permissions/987654321",
                    role=permission_services.to_role("reader"),
                    grantee_type=permission_services.to_grantee_type("everyone"),
                    email_address="_",
                ),
            ]

        @add_client_method
        def update_permission(
            request: glm.UpdatePermissionRequest,
        ) -> glm.Permission:
            self.observed_requests.append(request)
            return glm.Permission(
                name="corpora/demo-corpus/permissions/123456789",
                role=permission_services.to_role("reader"),
                grantee_type=permission_services.to_grantee_type("everyone"),
            )

        @add_client_method
        def transfer_ownership(
            request: glm.TransferOwnershipRequest,
        ) -> glm.TransferOwnershipResponse:
            self.observed_requests.append(request)
            return glm.TransferOwnershipResponse()

    def test_create_permission_success(self):
        x = retriever.create_corpus("demo-corpus")
        perm = x.create_permission(role="writer", grantee_type="everyone", email_address=None)
        self.assertIsInstance(perm, permission_services.Permission)
        self.assertIsInstance(self.observed_requests[-1], glm.CreatePermissionRequest)

    def test_create_permission_failure_email_set_when_grantee_type_is_everyone(self):
        x = retriever.create_corpus("demo-corpus")
        with self.assertRaises(ValueError):
            perm = x.create_permission(role="writer", grantee_type="everyone", email_address="_")

    def test_create_permission_failure_email_not_set_when_grantee_type_is_not_everyone(self):
        x = retriever.create_corpus("demo-corpus")
        with self.assertRaises(ValueError):
            perm = x.create_permission(role="writer", grantee_type="user", email_address=None)

    def test_delete_permission(self):
        x = retriever.create_corpus("demo-corpus")
        perm = x.create_permission("writer", "everyone")
        perm.delete()
        self.assertIsInstance(self.observed_requests[-1], glm.DeletePermissionRequest)

    def test_get_permission_with_full_name(self):
        x = retriever.create_corpus("demo-corpus")
        perm = x.create_permission("writer", "everyone")
        fetch_perm = permission.get_permission(name=perm.name)
        self.assertIsInstance(fetch_perm, permission_services.Permission)
        self.assertIsInstance(self.observed_requests[-1], glm.GetPermissionRequest)
        self.assertEqual(fetch_perm, perm)

    def test_get_permission_with_resource_name_and_id_1(self):
        x = retriever.create_corpus("demo-corpus")
        perm = x.create_permission("writer", "everyone")
        fetch_perm = permission.get_permission(
            resource_name="corpora/demo-corpus", permission_id=123456789
        )
        self.assertIsInstance(fetch_perm, permission_services.Permission)
        self.assertIsInstance(self.observed_requests[-1], glm.GetPermissionRequest)
        self.assertEqual(fetch_perm, perm)

    def test_get_permission_with_resource_name_name_and_id_2(self):
        fetch_perm = permission.get_permission(
            resource_name="tunedModels/demo-corpus", permission_id=123456789
        )
        self.assertIsInstance(fetch_perm, permission_services.Permission)
        self.assertIsInstance(self.observed_requests[-1], glm.GetPermissionRequest)

    def test_get_permission_with_resource_type(self):
        fetch_perm = permission.get_permission(
            resource_name="demo-model", permission_id=123456789, resource_type="tunedModels"
        )
        self.assertIsInstance(fetch_perm, permission_services.Permission)
        self.assertIsInstance(self.observed_requests[-1], glm.GetPermissionRequest)

    @parameterized.named_parameters(
        dict(
            testcase_name="no_information_provided",
        ),
        dict(
            testcase_name="permission_id_missing",
            resource_name="demo-corpus",
        ),
        dict(
            testcase_name="resource_name_missing",
            permission_id="123456789",
        ),
        dict(
            testcase_name="invalid_corpus_name", name="corpora/demo-corpus-/permissions/123456789"
        ),
        dict(testcase_name="invalid_permission_id", name="corpora/demo-corpus/permissions/*"),
        dict(
            testcase_name="invalid_tuned_model_name",
            name="tunedModels/my_text_model/permissions/123456789",
        ),
        dict(
            testcase_name="unsupported_resource_name_1",
            name="dataset/demo-corpus/permissions/123456789",
        ),
        dict(
            testcase_name="unsupported_resource_type_2",
            resource_name="my-dataset",
            permission_id="123456789",
            resource_type="dataset",
        ),
        dict(
            testcase_name="invlalid_full_name_format_1",
            name="corpora/demo-corpus/permissions/123456789/extra",
        ),
        dict(testcase_name="invlalid_full_name_format_2", name="corpora/2323"),
        dict(testcase_name="invlalid_full_name_format_3", name="corpora"),
    )
    def test_get_permission_with_invalid_name_constructs(
        self,
        name=None,
        resource_name=None,
        permission_id=None,
        resource_type=None,
    ):
        with self.assertRaises(ValueError):
            fetch_perm = permission.get_permission(
                name=name,
                resource_name=resource_name,
                permission_id=permission_id,
                resource_type=resource_type,
            )

    def test_list_permission(self):
        x = retriever.create_corpus("demo-corpus")
        perm1 = x.create_permission("writer", "everyone")
        perm2 = x.create_permission("reader", "group", "_")
        perms = list(x.list_permissions())
        self.assertEqual(len(perms), 2)
        self.assertEqual(perm1, perms[0])
        self.assertEqual(perms[1].email_address, "_")
        for perm in perms:
            self.assertIsInstance(perm, permission_services.Permission)
        self.assertIsInstance(self.observed_requests[-1], glm.ListPermissionsRequest)

    def test_update_permission_success(self):
        x = retriever.create_corpus("demo-corpus")
        perm = x.create_permission("writer", "everyone")
        updated_perm = perm.update({"role": permission_services.to_role("reader")})
        self.assertIsInstance(updated_perm, permission_services.Permission)
        self.assertIsInstance(self.observed_requests[-1], glm.UpdatePermissionRequest)

    def test_update_permission_failure_restricted_update_path(self):
        x = retriever.create_corpus("demo-corpus")
        perm = x.create_permission("writer", "everyone")
        with self.assertRaises(ValueError):
            updated_perm = perm.update(
                {"grantee_type": permission_services.to_grantee_type("user")}
            )

    def test_transfer_ownership(self):
        self.responses["get_tuned_model"] = glm.TunedModel(
            name="tunedModels/fake-pig-001", base_model="models/dance-monkey-007"
        )
        x = models.get_tuned_model("tunedModels/fake-pig-001")
        response = x.transfer_ownership(email_address="_")
        self.assertIsInstance(self.observed_requests[-1], glm.TransferOwnershipRequest)

    def test_transfer_ownership_on_corpora(self):
        x = retriever.create_corpus("demo-corpus")
        with self.assertRaises(NotImplementedError):
            x.transfer_ownership(email_address="_")

    @parameterized.named_parameters(
        [
            "create_permission",
            retriever_services.Corpus.create_permission,
            retriever_services.Corpus.create_permission_async,
        ],
        [
            "list_permissions",
            retriever_services.Corpus.list_permissions,
            retriever_services.Corpus.list_permissions_async,
        ],
        [
            "Permission.delete",
            permission_services.Permission.delete,
            permission_services.Permission.delete_async,
        ],
        [
            "Permission.update",
            permission_services.Permission.update,
            permission_services.Permission.update_async,
        ],
        [
            "Permission.get_permission",
            permission_services.Permission.get,
            permission_services.Permission.get_async,
        ],
        [
            "permission.get_permission",
            permission.get_permission,
            permission.get_permission_async,
        ],
        [
            "Permission.transfer_ownership",
            retriever_services.Corpus.transfer_ownership,
            retriever_services.Corpus.transfer_ownership_async,
        ],
        [
            "TunedModel.transfer_ownership",
            model_services.TunedModel.transfer_ownership,
            model_services.TunedModel.transfer_ownership_async,
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

    def test_create_corpus_called_with_request_options(self):
        self.client.create_corpus = unittest.mock.MagicMock()
        request = unittest.mock.ANY
        request_options = {"timeout": 120}

        try:
            retriever.create_corpus("demo-corpus", request_options=request_options)
        except AttributeError:
            pass

        self.client.create_corpus.assert_called_once_with(request, **request_options)


if __name__ == "__main__":
    absltest.main()
