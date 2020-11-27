# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/detectron2/utils/file_io.py
@Time         : 2020-11-26 12:51:43
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-27 20:30:47
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

import os
import qiniu

from fvcore.common.file_io import (
    PathHandler,
    PathManagerBase as _PathManagerBase,
    HTTPURLHandler,
    OneDrivePathHandler
)

__all__ = ["PathManager", "PathHandler"]


class Detectron2Handler(PathHandler):
    """
    Resolve anything that's hosted under detectron2's namespace.
    """

    PREFIX = "detectron2://"
    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX):]
        return PathManager.get_local_path(self.S3_DETECTRON2_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


class KODOHandler(PathHandler):
    """
    Resolve anything that's in KODO. (URL like KODO://)
    See: https://www.qiniu.com/products/kodo for more usage details.
    """

    PREFIX = "KODO://"
    KODO_PREFIX = os.getenv("kodo_prefix", "http://det.cjh.zone/detectron2/")

    access_key = os.getenv("kodo_access_key", None)
    secret_key = os.getenv("kodo_secret_key", None)
    if access_key is not None and secret_key is not None:
        AUTH = qiniu.Auth(access_key, secret_key)

    @classmethod
    def check_auth(cls):
        assert hasattr(cls, "AUTH"), (
            "The environment variables 'kodo_access_key' or 'kodo_secret_key' are not found."
        )

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX):]
        return PathManager.get_local_path(self.KODO_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)

    def _upload(self, local: str, remote: str, **kwargs) -> bool:
        self.check_auth()

        assert "bucket" in kwargs

        bucket = kwargs.pop("bucket")
        name = remote[len(self.PREFIX):]
        token = self.AUTH.upload_token(bucket, name)
        ret, info = qiniu.put_file(token, name, local)
        # assert ret['key'] == name
        # assert ret['hash'] == qiniu.etag(local)
        return info.status_code == 200


class PathManagerBase(_PathManagerBase):

    def upload(self, local: str, remote: str, **kwargs):
        """
        Upload the local file (not directory) to the specified remote URI.
        Args:
            local (str): path of the local file to be uploaded.
            remote (str): the remote uri.
        """
        handler = self.__get_path_handler(remote)
        assert isinstance(handler, KODOHandler), "Invalid remote path: {}.".format(remote)
        return handler._upload(local, remote, **kwargs)


PathManager = PathManagerBase()
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""

PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
PathManager.register_handler(Detectron2Handler())
PathManager.register_handler(KODOHandler())
