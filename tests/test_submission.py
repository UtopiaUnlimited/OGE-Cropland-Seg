import copy
import hashlib
import io
import json
import tarfile

import pytest

from scripts.package_submission import finalize_archive


@pytest.mark.parametrize("store", ["classic", "containerd"])
def test_saved_archive_checks_layers_and_runtime_not_image_id_shape(tmp_path, store):
    runtime = dict(Entrypoint=["python", "-m", "cropland.inference"], Cmd=None,
                   Env=["HF_HUB_OFFLINE=1"], WorkingDir="/app", User="", Labels={"purpose": "test"})
    config = dict(os="linux", architecture="amd64", config=runtime,
                  rootfs=dict(type="layers", diff_ids=["sha256:expected-layer"]))
    raw = json.dumps(config).encode()
    image_id = "sha256:" + (hashlib.sha256(raw).hexdigest() if store == "classic" else "a" * 64)
    inspected_runtime = runtime.copy()
    if store == "containerd":
        inspected_runtime.pop("User")
        inspected_runtime.pop("Cmd")
    image = dict(Id=image_id, RootFS=dict(Layers=config["rootfs"]["diff_ids"]), Config=inspected_runtime)
    validation = dict(image_id=image_id, image_tag="oge:test")
    archive = tmp_path / "model.tar"
    with tarfile.open(archive, "w") as tar:
        members = {"config.json": raw, "manifest.json": json.dumps(
            [{"Config": "config.json", "RepoTags": ["oge:test"], "Layers": []}]).encode()}
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    result = finalize_archive(archive, image, validation)
    assert result["archive"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    changed = copy.deepcopy(image)
    changed["Config"]["Entrypoint"] = ["unexpected-command"]
    with pytest.raises(ValueError, match="Entrypoint"):
        finalize_archive(archive, changed, validation)
    changed = copy.deepcopy(image)
    changed["RootFS"]["Layers"] = ["sha256:wrong-layer"]
    with pytest.raises(ValueError, match="filesystem layers"):
        finalize_archive(archive, changed, validation)
