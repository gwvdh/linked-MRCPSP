import xml.etree.ElementTree as ET
import argparse
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generator, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


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
    """Parsed directly from XML — one per <call> element."""
    call_id: str
    label: str
    leaves: list[Leaf] = field(default_factory=list)


@dataclass
class TreeNode:
    """One node in the expanded process tree (one call × one resource)."""
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
            f"[{self.call_id}] {self.call_label}"
            f" / {self.resource_id} {self.resource_name} (cost={self.cost})"
        )


def dummy_node(children: list[TreeNode]) -> TreeNode:
    return TreeNode(
        call_id="dummy", call_label="—",
        resource_id="—", resource_name="—",
        cost=0, is_dummy=True, children=children,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _ns(root: ET.Element) -> str:
    """Extract namespace prefix from root tag, e.g. '{http://...}'."""
    prefix, found, _ = root.tag.partition("}")
    return (prefix + "}") if found else ""
    #m = re.match(r"\{(.+?)\}", root.tag)
    #return f"{{{m.group(1)}}}" if m else ""


def _parse_leaf(resource: ET.Element, ns: str) -> Optional[Leaf]:
    resprofile = resource.find(f"{ns}resprofile")
    if resprofile is None:
        return None

    res_id = resource.get("id", "")
    res_name = resource.get("name", "")
    cost = int(resprofile.findtext(f"{ns}measures/{ns}cost", default="0"))
    cp_el = resprofile.find(f"{ns}changepattern")

    if cp_el is None:
        return Leaf(resource_id=res_id, resource_name=res_name, cost=cost)

    manipulate = (
        resprofile.find(f"{ns}children/{ns}manipulate")
        or cp_el.find(f".//{ns}manipulate")
    )

    inserted = [
        Leaf(
            resource_id=r.get("id", ""),
            resource_name=r.get("name", ""),
            cost=int(rp.findtext(f"{ns}measures/{ns}cost", default="0"))
            if (rp := r.find(f"{ns}resprofile")) is not None
            else 0,
        )
        for r in (
            manipulate.findall(f"{ns}children/{ns}resource")
            if manipulate is not None
            else []
        )
    ]

    return Leaf(
        resource_id=res_id,
        resource_name=res_name,
        cost=cost,
        cp_type=cp_el.get("type", ""),
        cp_direction=cp_el.findtext(f"{ns}parameters/{ns}direction", default=""),
        cp_target_label=manipulate.get("label", "") if manipulate is not None else "",
        cp_target_id=manipulate.get("id", "") if manipulate is not None else "",
        cp_inserted_leaves=inserted,
    )


def parse_calls(xml_file: str) -> list[CallNode]:
    root = ET.parse(xml_file).getroot()
    ns = _ns(root)
    calls = []
    for call in root.findall(f"{ns}call"):
        node = CallNode(
            call_id=call.get("id"),
            label=call.findtext(f"{ns}parameters/{ns}label", default=""),
        )
        for resource in call.findall(f"{ns}children/{ns}resource"):
            leaf = _parse_leaf(resource, ns)
            if leaf is not None:
                node.leaves.append(leaf)
        calls.append(node)
    return calls


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------


def build_tree(calls: list[CallNode]) -> list[TreeNode]:
    """
    Recursively expand calls into a tree of (call × resource) nodes.
    Change patterns are applied per-path:
      DELETE — skip the named call on this path
      INSERT — inject a synthetic call immediately after the current one
    """

    def attach(remaining: list[CallNode], skipped: set[str]) -> list[TreeNode]:
        if not remaining:
            return []

        current, *rest = remaining
        if current.label in skipped:
            return attach(rest, skipped)

        nodes = []
        for leaf in current.leaves:
            child_calls = list(rest)
            new_skipped = set(skipped)

            if leaf.cp_type == "delete":
                new_skipped.add(leaf.cp_target_label)
            elif leaf.cp_type == "insert":
                child_calls = [
                    CallNode(
                        call_id=leaf.cp_target_id,
                        label=leaf.cp_target_label,
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
                    children=attach(child_calls, new_skipped),
                )
            )

        return nodes

    return attach(calls, set())


# ---------------------------------------------------------------------------
# Tree equalisation
# ---------------------------------------------------------------------------


def _depth(node: TreeNode) -> int:
    return 1 if not node.children else 1 + max(_depth(c) for c in node.children)


def equalize(nodes: list[TreeNode]) -> list[TreeNode]:
    """
    Bottom-up: insert dummy nodes between a node and its children so that
    all siblings reach the same depth, aligning identical calls across paths.
    """
    if not nodes:
        return nodes

    for node in nodes:
        node.children = equalize(node.children)

    max_d = max(_depth(n) for n in nodes)
    for node in nodes:
        inner = node.children
        for _ in range(max_d - _depth(node)):
            inner = [dummy_node(inner)]
        node.children = inner

    return nodes


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def get_leaves(nodes: list[TreeNode]) -> Generator[TreeNode, None, None]:
    """Yield all nodes with no children (lowest-level leaves)."""
    for node in nodes:
        if not node.children:
            yield node
        else:
            yield from get_leaves(node.children)


def get_paths(nodes: list[TreeNode]) -> list[list[TreeNode]]:
    """
    Return all root-to-leaf paths as lists of TreeNodes.
    Since the tree is already equalized, all paths are the same length.
    """
    paths: list[list[TreeNode]] = []

    def traverse(node: TreeNode, path: list[TreeNode]) -> None:
        path = path + [node]
        if not node.children:
            paths.append(path)
        else:
            for child in node.children:
                traverse(child, path)

    for root in nodes:
        traverse(root, [])

    return paths


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_tree(nodes: list[TreeNode], prefix: str = "") -> None:
    for i, node in enumerate(nodes):
        last = i == len(nodes) - 1
        print(f"{prefix}{'└──' if last else '├──'} {node}")
        print_tree(node.children, prefix + ("    " if last else "│   "))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


class RA_PST:
    def __init__(self, xml_file: str):
        self.xml_file = xml_file
        self.calls = parse_calls(xml_file)
        self.roots = equalize(build_tree(self.calls))
        self.paths = get_paths(self.roots)

    def get_number_of_tasks(self) -> int:
        return len(self.paths[0])

    def get_number_of_modes(self) -> int:
        return len(self.paths)

    def get_resource(self, task: int, mode: int) -> Optional[int]:
        ids = self.resource_ids
        rid = self.paths[mode][task].resource_id
        return ids.index(rid) if rid in ids else None

    @cached_property
    def resource_ids(self) -> list[str]:
        """Sorted list of all real (non-dummy) resource IDs across all paths."""
        ids = {
            leaf.resource_id
            for path in self.paths
            for leaf in path
            if leaf.resource_id != "—"
        }
        return sorted(ids)

    def get_resource_ids(self) -> list[str]:
        return self.resource_ids

    def get_number_of_resources(self) -> int:
        return len(self.resource_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Parse a CPEE XML description into an expanded process tree."
    )
    parser.add_argument("xml_file", help="Path to the XML description file")
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
    print(f"resource of task 3 in mode 1: {ra_pst.get_resource(3, 1)}")


if __name__ == "__main__":
    main()
