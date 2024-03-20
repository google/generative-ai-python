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
from typing import Any
import unittest
import unittest.mock as mock

import google.ai.generativelanguage as glm

from google.generativeai import retriever
from google.generativeai import permission
from google.generativeai.types import permission_types as permission_services
from google.generativeai.types import retriever_types as retriever_services

from google.generativeai import client
from absl.testing import absltest
from absl.testing import parameterized


class UnitTests(parameterized.TestCase):
    def setUp(self):
        self.client = unittest.mock.MagicMock()

        client._client_manager.clients["retriever"] = self.client
        client._client_manager.clients["permission"] = self.client

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
                name="corpora/demo_corpus",
                display_name="demo-corpus",
                create_time="2000-01-01T01:01:01.123456Z",
                update_time="2000-01-01T01:01:01.123456Z",
            )

        @add_client_method
        def create_permission(
            request: glm.CreatePermissionRequest,
        ) -> glm.Permission:
            self.observed_requests.append(request)
            return glm.Permission(
                name="corpora/demo_corpus/permissions/123456789",
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
                name="corpora/demo_corpus/permissions/123456789",
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
                    name="corpora/demo_corpus/permissions/123456789",
                    role=permission_services.to_role("writer"),
                    grantee_type=permission_services.to_grantee_type("everyone"),
                ),
                glm.Permission(
                    name="corpora/demo_corpus/permissions/987654321",
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
                name="corpora/demo_corpus/permissions/123456789",
                role=permission_services.to_role("reader"),
                grantee_type=permission_services.to_grantee_type("everyone"),
            )

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

    def test_get_permission(self):
        x = retriever.create_corpus("demo-corpus")
        perm = x.create_permission("writer", "everyone")
        fetch_perm = permission.get_permission(name=perm.name)
        self.assertIsInstance(fetch_perm, permission_services.Permission)
        self.assertIsInstance(self.observed_requests[-1], glm.GetPermissionRequest)
        self.assertEqual(fetch_perm, perm)

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
