"""Explicit destination for analysis records, images and arrays."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from pemap.io.images import png
from pemap.io.files import sha, write


@dataclass(frozen=True)
class AnalysisOutput:
    """An explicit output destination; never inspects a scene runner's state."""

    directory: Path
    published_directory: Path

    def record(self, name, value):
        write(self.directory / name, value)

    def image(self, name, value):
        png(self.directory / name, value)

    def array(self, name, value):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value)
        return {"path": str(self.published_directory / name), "sha256": sha(path)}
