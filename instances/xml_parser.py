from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generator, Optional


# ---------------------------------------------------------------------
# Parsed XML entities
# ---------------------------------------------------------------------


@dataclass
class Leaf:
    resource_id: str
    resource_name: str
    cost: int
    cp_type: Optional[str] = None
    cp_direction: Optional[str] = None
    cp_target_label: Optional[str] = None
    cp_target_id: Optional[str] = None
    cp_inserted_leaves: list["Leaf"] = field(default_factory=list)


@dataclass
class CallNode:
    call_id: str
    label: str
    leaves: list[Leaf] = field(default_factory=list)


@dataclass
class TreeNode:
    call_id: str
    call_label: str
    resource_id: str
    resource_name: str
    cost: int
    is_dummy: bool = False
    children: list["TreeNode"] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_dummy:
            return "[dummy]"
        return (
            f"[{self.call_id}] {self.call_label} / "
            f"{self.resource_id} {self.resource_name} (cost={self.cost})"
        )


def make_dummy(children: list[TreeNode]) -> TreeNode:
    return TreeNode(
        call_id="dummy",
        call_label="—",
        resource_id="—",
        resource_name="—",
        cost=0,
        is_dummy=True,
        children=children,
    )


# ---------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------


def _namespace(root: ET.Element) -> str:
    prefix, found, _ = root.tag.partition("}")
    return f"{prefix}}}" if found else ""


def _parse_leaf(resource_el: ET.Element, ns: str) -> Optional[Leaf]:
    resprofile = resource_el.find(f"{ns}resprofile")
    if resprofile is None:
        return None

    resource_id = resource_el.get("id", "")
    resource_name = resource_el.get("name", "")
    cost = int(resprofile.findtext(f"{ns}measures/{ns}cost", default="0"))

    cp_el = resprofile.find(f"{ns}changepattern")
    if cp_el is None:
        return Leaf(resource_id=resource_id, resource_name=resource_name, cost=cost)

    manipulate = (
        resprofile.find(f"{ns}children/{ns}manipulate")
        or cp_el.find(f".//{ns}manipulate")
    )

    inserted_leaves: list[Leaf] = []
    if manipulate is not None:
        for r in manipulate.findall(f"{ns}children/{ns}resource"):
            rp = r.find(f"{ns}resprofile")
            inserted_leaves.append(
                Leaf(
                    resource_id=r.get("id", ""),
                    resource_name=r.get("name", ""),
                    cost=int(
                        rp.findtext(f"{ns}measures/{ns}cost", default="0")
                        if rp is not None
                        else 0
                    ),
                )
            )

    return Leaf(
        resource_id=resource_id,
        resource_name=resource_name,
        cost=cost,
        cp_type=cp_el.get("type", ""),
        cp_direction=cp_el.findtext(f"{ns}parameters/{ns}direction", default=""),
        cp_target_label=manipulate.get("label", "") if manipulate is not None else "",
        cp_target_id=manipulate.get("id", "") if manipulate is not None else "",
        cp_inserted_leaves=inserted_leaves,
    )


def parse_calls(xml_file: str) -> list[CallNode]:
    root = ET.parse(xml_file).getroot()
    ns = _namespace(root)

    calls: list[CallNode] = []
    for call_el in root.findall(f"{ns}call"):
        call = CallNode(
            call_id=call_el.get("id", ""),
            label=call_el.findtext(f"{ns}parameters/{ns}label", default=""),
        )
        for resource_el in call_el.findall(f"{ns}children/{ns}resource"):
            leaf = _parse_leaf(resource_el, ns)
            if leaf is not None:
                call.leaves.append(leaf)
        calls.append(call)

    return calls


# ---------------------------------------------------------------------
# Tree construction and alignment
# ---------------------------------------------------------------------


def build_tree(calls: list[CallNode]) -> list[TreeNode]:
    """
    Expand the XML description into a tree whose root-to-leaf paths define modes.
    Change patterns are applied path-wise:
      - delete: skip target call on that path
      - insert: inject a synthetic call immediately after the current one
    """

    def expand(remaining: list[CallNode], skipped: set[str]) -> list[TreeNode]:
        if not remaining:
            return []

        current, *rest = remaining
        if current.label in skipped:
            return expand(rest, skipped)

        nodes: list[TreeNode] = []
        for leaf in current.leaves:
            child_calls = list(rest)
            skipped_now = set(skipped)

            if leaf.cp_type == "delete" and leaf.cp_target_label:
                skipped_now.add(leaf.cp_target_label)
            elif leaf.cp_type == "insert":
                child_calls = [
                    CallNode(
                        call_id=leaf.cp_target_id or "",
                        label=leaf.cp_target_label or "",
                        leaves=leaf.cp_inserted_leaves,
                    )
                ] + child_calls

            nodes.append(
                TreeNode(
                    call_id=current.call_id,
                    call_label=current.label,
                    resource_id=leaf.resource_id,
                    resource_name=leaf.resource_name,
                    cost=leaf.cost,
                    children=expand(child_calls, skipped_now),
                )
            )
        return nodes

    return expand(calls, set())


def _depth(node: TreeNode) -> int:
    if not node.children:
        return 1
    return 1 + max(_depth(child) for child in node.children)


def equalize(nodes: list[TreeNode]) -> list[TreeNode]:
    """
    Insert dummy nodes so that all root-to-leaf paths have equal depth.
    """
    if not nodes:
        return nodes

    for node in nodes:
        node.children = equalize(node.children)

    target_depth = max(_depth(node) for node in nodes)
    for node in nodes:
        children = node.children
        for _ in range(target_depth - _depth(node)):
            children = [make_dummy(children)]
        node.children = children

    return nodes


def get_leaves(nodes: list[TreeNode]) -> Generator[TreeNode, None, None]:
    for node in nodes:
        if not node.children:
            yield node
        else:
            yield from get_leaves(node.children)


def get_paths(nodes: list[TreeNode]) -> list[list[TreeNode]]:
    paths: list[list[TreeNode]] = []

    def dfs(node: TreeNode, prefix: list[TreeNode]) -> None:
        path = prefix + [node]
        if not node.children:
            paths.append(path)
            return
        for child in node.children:
            dfs(child, path)

    for root in nodes:
        dfs(root, [])

    return paths


def print_tree(nodes: list[TreeNode], prefix: str = "") -> None:
    for idx, node in enumerate(nodes):
        last = idx == len(nodes) - 1
        print(f"{prefix}{'└──' if last else '├──'} {node}")
        print_tree(node.children, prefix + ("    " if last else "│   "))


# ---------------------------------------------------------------------
# RA-PST wrapper
# ---------------------------------------------------------------------


class RA_PST:
    """
    Canonical process-tree representation used by the instance generator.
    Paths correspond to modes; positions in a path correspond to tasks.
    """

    def __init__(
        self,
        xml_file: str,
        global_resource_ids: Optional[list[str]] = None,
    ):
        self.xml_file = xml_file
        self.calls = parse_calls(xml_file)
        self.roots = equalize(build_tree(self.calls))
        self.paths = get_paths(self.roots)
        self._global_resource_ids = global_resource_ids

    def get_number_of_tasks(self) -> int:
        return len(self.paths[0])

    def get_number_of_modes(self) -> int:
        return len(self.paths)

    @cached_property
    def resource_ids(self) -> list[str]:
        return sorted(
            {
                node.resource_id
                for path in self.paths
                for node in path
                if node.resource_id != "—"
            }
        )

    def get_resource_ids(self) -> list[str]:
        return (
            self._global_resource_ids
            if self._global_resource_ids is not None
            else self.resource_ids
        )

    def get_number_of_resources(self) -> int:
        return len(self.get_resource_ids())

    def get_resource(self, task: int, mode: int) -> Optional[int]:
        resource_ids = self.get_resource_ids()
        rid = self.paths[mode][task].resource_id
        return resource_ids.index(rid) if rid in resource_ids else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse an XML RA-PST into an expanded process tree."
    )
    parser.add_argument("xml_file", help="Path to the XML file")
    args = parser.parse_args()

    ra_pst = RA_PST(args.xml_file)

    print("Process Tree")
    print("=" * 60)
    print_tree(ra_pst.roots)

    print("\nPaths")
    print("=" * 60)
    for path in ra_pst.paths:
        print(" ->\n  ".join(str(node) for node in path))
        print()

    print(f"number of tasks: {ra_pst.get_number_of_tasks()}")
    print(f"number of modes: {ra_pst.get_number_of_modes()}")
    print(f"resource ids: {ra_pst.get_resource_ids()}")
    print(f"number of resources: {ra_pst.get_number_of_resources()}")


if __name__ == "__main__":
    main()
