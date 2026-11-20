"""Read existing submission archives in an isolated solutions namespace.

Only absolute solution import names are redirected. Controller source bytes,
engine imports and resources remain those recorded by the archive/audit. The
workspace entry, modules and archive are never changed. Missing archive members
fail closed; a workspace solution with the same filename cannot satisfy them.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import io
from pathlib import Path
import sys
import zipfile

import yaml


SOLUTION_PREFIX = "swarm_rescue.solutions."
_CONTEXTS = {}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class _SolutionImports(ast.NodeTransformer):
    def __init__(self, namespace):
        self.namespace = namespace

    def visit_ImportFrom(self, node):
        if node.level:
            raise ValueError("Relative solution imports require explicit review")
        if node.module == "swarm_rescue.solutions":
            node.module = self.namespace
        elif (node.module or "").startswith(SOLUTION_PREFIX):
            node.module = self.namespace + "." + node.module[len(SOLUTION_PREFIX):]
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith(SOLUTION_PREFIX):
                if not alias.asname:
                    raise ValueError("Unaliased dotted solution imports require explicit review")
                alias.name = self.namespace + "." + alias.name[len(SOLUTION_PREFIX):]
        return node


class _ArchiveContext(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        archive_bytes = self.path.read_bytes()
        self.sha256 = _sha(archive_bytes)
        self.namespace = "_swarm_red_archive_" + self.sha256
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise ValueError("Duplicate archive members")
            if archive.testzip() is not None:
                raise ValueError("Archive CRC check failed")
            self.files = {name: archive.read(name) for name in names if not name.endswith("/")}
        for name in self.files:
            if "\\" in name or name.startswith("/") or ".." in name.split("/"):
                raise ValueError("Unsafe archive member: " + name)
        for name in ("my_drone_eval.py", "team_info.yml", "req.txt", "__init__.py"):
            if name not in self.files:
                raise ValueError("Required root archive member absent: " + name)
        self.bytes = len(archive_bytes)
        self.loaded = {}

    def _member(self, fullname):
        if fullname == self.namespace:
            return "__init__.py", True
        relative = fullname[len(self.namespace) + 1:].replace(".", "/")
        if relative + ".py" in self.files:
            return relative + ".py", False
        if relative + "/__init__.py" in self.files:
            return relative + "/__init__.py", True
        raise ModuleNotFoundError("Missing archive dependency; workspace fallback forbidden: " + fullname)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.namespace and not fullname.startswith(self.namespace + "."):
            return None
        member, is_package = self._member(fullname)
        return importlib.util.spec_from_loader(fullname, self, is_package=is_package,
                                               origin=str(self.path) + "!/" + member)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        member, _ = self._member(module.__name__)
        source = self.files[member]
        module.__file__ = str(self.path) + "!/" + member
        tree = ast.parse(source, filename=module.__file__)
        # These archives have statically resolvable imports. Do not silently
        # permit a future archive to escape isolation through dynamic imports.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "__import__"):
                    raise ValueError("Dynamic execution/import requires archive review: " + member)
                if isinstance(node.func, ast.Attribute) and node.func.attr in ("import_module", "reload"):
                    raise ValueError("Dynamic import requires archive review: " + member)
        tree = _SolutionImports(self.namespace).visit(tree)
        ast.fix_missing_locations(tree)
        exec(compile(tree, module.__file__, "exec"), module.__dict__)
        self.loaded[member] = _sha(source)

    def closure(self, start):
        pending = [start]
        result, external = {}, set()
        while pending:
            name = pending.pop()
            member = name.replace(".", "/") + ".py"
            if member in result:
                continue
            if member not in self.files:
                raise ModuleNotFoundError("Missing archive dependency: " + member)
            source = self.files[member]
            result[member] = _sha(source)
            for node in ast.walk(ast.parse(source, filename=member)):
                imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                for dependency in imports:
                    if dependency.startswith(SOLUTION_PREFIX):
                        pending.append(dependency[len(SOLUTION_PREFIX):])
                    elif dependency:
                        external.add(dependency)
        return result, sorted(external)


def load_archived_controller(zip_path):
    """Return the class selected by this exact archive's root PLACE factory."""
    context = _ArchiveContext(zip_path)
    if context.namespace in _CONTEXTS:
        context = _CONTEXTS[context.namespace]
    else:
        _CONTEXTS[context.namespace] = context
        sys.meta_path.insert(0, context)
    entry = importlib.import_module(context.namespace + ".my_drone_eval")
    controller = entry.drone_class_for_mode("place")
    if not controller.__module__.startswith(context.namespace + "."):
        raise ValueError("Archive PLACE factory returned a non-archive controller")
    return controller


def archive_runtime_identity(controller):
    """JSON-safe byte provenance for the actual loaded archive controller."""
    namespace = controller.__module__.split(".", 1)[0]
    context = _CONTEXTS[namespace]
    original_module = SOLUTION_PREFIX + controller.__module__[len(namespace) + 1:]
    return dict(zip_path=str(context.path), zip_sha256=context.sha256,
                controller_module=original_module, controller_class=controller.__name__,
                isolated_module=controller.__module__,
                loaded_member_sha256=dict(sorted(context.loaded.items())),
                loaded_member_origins={member: str(context.path) + "!/" + member
                                       for member in sorted(context.loaded)},
                workspace_solution_fallback_allowed=False,
                source_transform="absolute solutions import namespace only")


def audit_archive(zip_path):
    """Audit actual factory result, dependency closure, metadata and member hashes."""
    controller = load_archived_controller(zip_path)
    runtime = archive_runtime_identity(controller)
    context = _CONTEXTS[controller.__module__.split(".", 1)[0]]
    member_module = controller.__module__[len(context.namespace) + 1:]
    closure, external = context.closure(member_module)
    entry = sys.modules[context.namespace + ".my_drone_eval"]
    try:
        rescue = entry.drone_class_for_mode("rescue")
        rescue_identity = rescue.__module__.replace(context.namespace + ".", SOLUTION_PREFIX) + "." + rescue.__name__
    except ValueError as exc:
        rescue_identity = "REJECTED: " + str(exc)
    return dict(zip_path=str(Path(zip_path).resolve()), zip_sha256=context.sha256,
                zip_bytes=context.bytes, crc_ok=True,
                entries={name: dict(bytes=len(data), sha256=_sha(data))
                         for name, data in sorted(context.files.items())},
                controller=runtime["controller_module"] + "." + runtime["controller_class"],
                controller_module=runtime["controller_module"], controller_class=runtime["controller_class"],
                rescue=rescue_identity, red_dependency_sha256=closure,
                external_imports=external,
                entry_text=context.files["my_drone_eval.py"].decode("utf-8"),
                team_info=yaml.safe_load(context.files["team_info.yml"]),
                requirements_text=context.files["req.txt"].decode("utf-8"),
                runtime=runtime)
