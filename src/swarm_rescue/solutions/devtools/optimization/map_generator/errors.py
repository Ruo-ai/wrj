"""Recoverable per-seed generation failures, separate from loading or I/O errors."""


class LayoutExhaustedError(ValueError):
    pass


class DuplicateStructureError(FileExistsError):
    pass
