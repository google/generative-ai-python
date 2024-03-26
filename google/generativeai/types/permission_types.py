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
from __future__ import annotations

import dataclasses
from typing import Optional, Union, Any, Iterable, AsyncIterable
import re

import google.ai.generativelanguage as glm

from google.protobuf import field_mask_pb2

from google.generativeai.client import get_dafault_permission_client
from google.generativeai.client import get_dafault_permission_async_client
from google.generativeai.utils import flatten_update_paths
from google.generativeai import string_utils


GranteeType = glm.Permission.GranteeType
Role = glm.Permission.Role

GranteeTypeOptions = Union[str, int, GranteeType]
RoleOptions = Union[str, int, Role]

_GRANTEE_TYPE: dict[GranteeTypeOptions, GranteeType] = {
    GranteeType.GRANTEE_TYPE_UNSPECIFIED: GranteeType.GRANTEE_TYPE_UNSPECIFIED,
    0: GranteeType.GRANTEE_TYPE_UNSPECIFIED,
    "grantee_type_unspecified": GranteeType.GRANTEE_TYPE_UNSPECIFIED,
    "unspecified": GranteeType.GRANTEE_TYPE_UNSPECIFIED,
    GranteeType.USER: GranteeType.USER,
    1: GranteeType.USER,
    "user": GranteeType.USER,
    GranteeType.GROUP: GranteeType.GROUP,
    2: GranteeType.GROUP,
    "group": GranteeType.GROUP,
    GranteeType.EVERYONE: GranteeType.EVERYONE,
    3: GranteeType.EVERYONE,
    "everyone": GranteeType.EVERYONE,
}

_ROLE: dict[RoleOptions, Role] = {
    Role.ROLE_UNSPECIFIED: Role.ROLE_UNSPECIFIED,
    0: Role.ROLE_UNSPECIFIED,
    "role_unspecified": Role.ROLE_UNSPECIFIED,
    "unspecified": Role.ROLE_UNSPECIFIED,
    Role.OWNER: Role.OWNER,
    1: Role.OWNER,
    "owner": Role.OWNER,
    Role.WRITER: Role.WRITER,
    2: Role.WRITER,
    "writer": Role.WRITER,
    Role.READER: Role.READER,
    3: Role.READER,
    "reader": Role.READER,
}

_VALID_PERMISSION_ID = r"permissions/([a-z0-9]+)$"
INVALID_PERMISSION_ID_MSG = """`permission_id` must follow the pattern: `permissions/<id>` and must \
consist of only alphanumeric characters. Got: `{permission_id}` instead."""


def to_grantee_type(x: GranteeTypeOptions) -> GranteeType:
    if isinstance(x, str):
        x = x.lower()
    return _GRANTEE_TYPE[x]


def to_role(x: RoleOptions) -> Role:
    if isinstance(x, str):
        x = x.lower()
    return _ROLE[x]


def valid_id(name: str) -> bool:
    return re.match(_VALID_PERMISSION_ID, name) is not None


@string_utils.prettyprint
@dataclasses.dataclass
class Permission:
    """
    A permission to access a resource.
    """

    name: str
    role: Role
    grantee_type: Optional[GranteeType]
    email_address: Optional[str] = None

    def delete(
        self,
        client: glm.PermissionServiceClient | None = None,
    ) -> None:
        """
        Delete permission (self).
        """
        if client is None:
            client = get_dafault_permission_client()
        delete_request = glm.DeletePermissionRequest(name=self.name)
        client.delete_permission(request=delete_request)

    async def delete_async(
        self,
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> None:
        """
        This is the async version of `Permission.delete`.
        """
        if client is None:
            client = get_dafault_permission_async_client()
        delete_request = glm.DeletePermissionRequest(name=self.name)
        await client.delete_permission(request=delete_request)

    # TODO (magashe): Add a method to validate update value. As of now only `role` is supported as a mask path
    def _apply_update(self, path, value):
        parts = path.split(".")
        for part in parts[:-1]:
            self = getattr(self, part)
        setattr(self, parts[-1], value)

    def update(
        self,
        updates: dict[str, Any],
        client: glm.PermissionServiceClient | None = None,
    ) -> Permission:
        """
        Update a list of fields for a specified permission.

        Args:
            updates: The list of fields to update.
                     Currently only `role` is supported as an update path.

        Returns:
            `Permission` object with specified updates.
        """
        if client is None:
            client = get_dafault_permission_client()

        updates = flatten_update_paths(updates)
        for update_path in updates:
            if update_path != "role":
                raise ValueError(
                    f"As of now, only `role` can be updated for `Permission`. Got: `{update_path}` instead."
                )
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        update_request = glm.UpdatePermissionRequest(
            permission=self._to_proto(), update_mask=field_mask
        )
        client.update_permission(request=update_request)
        return self

    async def update_async(
        self,
        updates: dict[str, Any],
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> Permission:
        """
        This is the async version of `Permission.update`.
        """
        if client is None:
            client = get_dafault_permission_async_client()

        updates = flatten_update_paths(updates)
        for update_path in updates:
            if update_path != "role":
                raise ValueError(
                    f"As of now, only `role` can be updated for `Permission`. Got: `{update_path}` instead."
                )
        field_mask = field_mask_pb2.FieldMask()

        for path in updates.keys():
            field_mask.paths.append(path)
        for path, value in updates.items():
            self._apply_update(path, value)

        update_request = glm.UpdatePermissionRequest(
            permission=self._to_proto(), update_mask=field_mask
        )
        await client.update_permission(request=update_request)
        return self

    def _to_proto(self) -> glm.Permission:
        return glm.Permission(
            name=self.name,
            role=self.role,
            grantee_type=self.grantee_type,
            email_address=self.email_address,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def get(
        cls,
        name: str,
        client: glm.PermissionServiceClient | None = None,
    ) -> Permission:
        """
        Get information about a specific permission.

        Args:
            name: The name of the permission to get.

        Returns:
            Requested permission as an instance of `Permission`.
        """
        if client is None:
            client = get_dafault_permission_client()
        get_perm_request = glm.GetPermissionRequest(name=name)
        get_perm_response = client.get_permission(request=get_perm_request)
        get_perm_response = type(get_perm_response).to_dict(get_perm_response)
        return cls(**get_perm_response)

    @classmethod
    async def get_async(
        cls,
        name: str,
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> Permission:
        """
        This is the async version of `Permission.get`.
        """
        if client is None:
            client = get_dafault_permission_async_client()
        get_perm_request = glm.GetPermissionRequest(name=name)
        get_perm_response = await client.get_permission(request=get_perm_request)
        get_perm_response = type(get_perm_response).to_dict(get_perm_response)
        return cls(**get_perm_response)


class PermissionAdapter:

    def _make_create_permission_request(
        self,
        role: RoleOptions,
        grantee_type: Optional[GranteeTypeOptions] = None,
        email_address: Optional[str] = None,
    ) -> glm.CreatePermissionRequest:
        role = to_role(role)

        if grantee_type:
            grantee_type = to_grantee_type(grantee_type)

        if email_address and grantee_type == GranteeType.EVERYONE:
            raise ValueError(
                f"Cannot limit access for: `{email_address}` when `grantee_type` is set to `EVERYONE`."
            )

        if not email_address and grantee_type != GranteeType.EVERYONE:
            raise ValueError(
                f"`email_address` must be specified unless `grantee_type` is set to `EVERYONE`."
            )

        permission = glm.Permission(
            role=role,
            grantee_type=grantee_type,
            email_address=email_address,
        )
        return glm.CreatePermissionRequest(
            parent=self.name,  # pytype: disable=attribute-error
            permission=permission,
        )

    def create_permission(
        self,
        role: RoleOptions,
        grantee_type: Optional[GranteeTypeOptions] = None,
        email_address: Optional[str] = None,
        client: glm.PermissionServiceClient | None = None,
    ) -> Permission:
        """
        Create a new permission on a resource (self).

        Args:
            parent: The resource name of the parent resource in which the permission will be listed.
            role: role that will be granted by the permission.
            grantee_type: The type of the grantee for the permission.
            email_address: The email address of the grantee.

        Returns:
            `Permission` object with specified parent, role, grantee type, and email address.

        Raises:
            ValueError: When email_address is specified and grantee_type is set to EVERYONE.
            ValueError: When email_address is not specified and grantee_type is not set to EVERYONE.
        """
        if client is None:
            client = get_dafault_permission_client()

        request = self._make_create_permission_request(
            role=role, grantee_type=grantee_type, email_address=email_address
        )
        permission_response = client.create_permission(request=request)
        permission_response = type(permission_response).to_dict(permission_response)
        return Permission(**permission_response)

    async def create_permission_async(
        self,
        role: RoleOptions,
        grantee_type: Optional[GranteeTypeOptions] = None,
        email_address: Optional[str] = None,
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> Permission:
        """
        This is the async version of `PermissionAdapter.create_permission`.
        """
        if client is None:
            client = get_dafault_permission_async_client()

        request = self._make_create_permission_request(
            role=role, grantee_type=grantee_type, email_address=email_address
        )
        permission_response = await client.create_permission(request=request)
        permission_response = type(permission_response).to_dict(permission_response)
        return Permission(**permission_response)

    def list_permissions(
        self,
        page_size: Optional[int] = None,
        client: glm.PermissionServiceClient | None = None,
    ) -> Iterable[Permission]:
        """
        List `Permission`s enforced on a resource (self).

        Args:
            parent: The resource name of the parent resource in which the permission will be listed.
            page_size: The maximum number of permissions to return (per page). The service may return fewer permissions.

        Returns:
            Paginated list of `Permission` objects.
        """
        if client is None:
            client = get_dafault_permission_client()

        request = glm.ListPermissionsRequest(
            parent=self.name, page_size=page_size  # pytype: disable=attribute-error
        )
        for permission in client.list_permissions(request):
            permission = type(permission).to_dict(permission)
            yield Permission(**permission)

    async def list_permissions_async(
        self,
        page_size: Optional[int] = None,
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> AsyncIterable[Permission]:
        """
        This is the async version of `PermissionAdapter.list_permissions`.
        """
        if client is None:
            client = get_dafault_permission_async_client()

        request = glm.ListPermissionsRequest(
            parent=self.name, page_size=page_size  # pytype: disable=attribute-error
        )
        async for permission in await client.list_permissions(request):
            permission = type(permission).to_dict(permission)
            yield Permission(**permission)

    def transfer_ownership(
        self,
        email_address: str,
        client: glm.PermissionServiceClient | None = None,
    ) -> None:
        """
        Transfer ownership of a resource (self) to a new owner.

        Args:
            name: Name of the resource to transfer ownership.
            email_address: Email address of the new owner.
        """
        if client is None:
            client = get_dafault_permission_client()
        transfer_request = glm.TransferOwnershipRequest(
            name=self.name, email_address=email_address  # pytype: disable=attribute-error
        )
        transfer_respone = client.transfer_ownership(request=transfer_request)

    async def transfer_ownership_async(
        self,
        email_address: str,
        client: glm.PermissionServiceAsyncClient | None = None,
    ) -> None:
        """This is the async version of `PermissionAdapter.transfer_ownership`."""
        if client is None:
            client = get_dafault_permission_async_client()
        transfer_request = glm.TransferOwnershipRequest(
            name=self.name, email_address=email_address  # pytype: disable=attribute-error
        )
        transfer_respone = await client.transfer_ownership(request=transfer_request)
