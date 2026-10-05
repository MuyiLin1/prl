from __future__ import annotations

from typing import Union

from prog_policies.base import BaseDSL, dsl_nodes
from prog_policies.base.dsl import _find_close_token


class SokobanDSL(BaseDSL):
    """DSL for the Sokoban domain.

    Mirrors ``MinigridDSL`` / ``KarelDSL`` but uses only plain boolean-feature
    perceptions (no parametrised feature nodes), so the generic
    ``ProgrammaticSpace`` drives it without a domain-specific subclass."""

    def __init__(self):
        nodes_list = [
            dsl_nodes.Repeat(),
            dsl_nodes.Concatenate(),
            dsl_nodes.Action("move"),
            dsl_nodes.Action("turnLeft"),
            dsl_nodes.Action("turnRight"),
            dsl_nodes.Action("solveStep"),
            dsl_nodes.BoolFeature("frontIsClear"),
            dsl_nodes.BoolFeature("leftIsClear"),
            dsl_nodes.BoolFeature("rightIsClear"),
            dsl_nodes.BoolFeature("frontIsBox"),
            dsl_nodes.BoolFeature("frontIsTarget"),
            dsl_nodes.BoolFeature("boxIsPushable"),
        ] + [dsl_nodes.ConstInt(i) for i in range(20)]
        super().__init__(nodes_list)

    @property
    def prod_rules(self):
        statements = [
            dsl_nodes.While,
            dsl_nodes.Repeat,
            dsl_nodes.If,
            dsl_nodes.ITE,
            dsl_nodes.Concatenate,
            dsl_nodes.Action,
        ]
        booleans = [
            dsl_nodes.BoolFeature,
            dsl_nodes.Not,
        ]
        statements_without_concat = [
            dsl_nodes.While,
            dsl_nodes.Repeat,
            dsl_nodes.If,
            dsl_nodes.ITE,
            dsl_nodes.Action,
        ]
        booleans_without_not = [
            dsl_nodes.BoolFeature,
        ]

        return {
            dsl_nodes.Program: [statements],
            dsl_nodes.While: [booleans, statements],
            dsl_nodes.Repeat: [[dsl_nodes.ConstInt], statements],
            dsl_nodes.If: [booleans, statements],
            dsl_nodes.ITE: [booleans_without_not, statements, statements],
            dsl_nodes.Concatenate: [statements_without_concat, statements],
            dsl_nodes.Not: [booleans_without_not],
        }

    def get_dsl_nodes_probs(self, node_type) -> dict[Union[str, dsl_nodes.BaseNode], float]:
        if node_type == dsl_nodes.StatementNode:
            return {
                dsl_nodes.While: 0.15,
                dsl_nodes.Repeat: 0.03,
                dsl_nodes.Concatenate: 0.5,
                dsl_nodes.If: 0.08,
                dsl_nodes.ITE: 0.04,
                dsl_nodes.Action: 0.2,
            }
        elif node_type == dsl_nodes.BoolNode:
            return {dsl_nodes.BoolFeature: 0.5, dsl_nodes.Not: 0.5}
        elif node_type == dsl_nodes.BoolFeature:
            return {name: 1.0 / len(self.bool_features) for name in self.bool_features}
        elif node_type == dsl_nodes.IntNode:
            return {dsl_nodes.ConstInt: 1.0}
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    @property
    def action_probs(self):
        return {
            "move": 0.4,
            "turnLeft": 0.1,
            "turnRight": 0.1,
            "solveStep": 0.4,
        }

    @property
    def bool_feat_probs(self):
        return {
            "frontIsClear": 0.25,
            "leftIsClear": 0.15,
            "rightIsClear": 0.15,
            "frontIsBox": 0.2,
            "frontIsTarget": 0.1,
            "boxIsPushable": 0.15,
        }

    @property
    def const_int_probs(self):
        return {i: 1 / 20 for i in range(20)}

    # ------------------------------------------------------------ to string
    def parse_node_to_str(self, node: dsl_nodes.BaseNode) -> str:
        if node is None:
            return "<HOLE>"

        if isinstance(node, dsl_nodes.ConstInt):
            return "R=" + str(node.value)
        if isinstance(node, dsl_nodes.ConstBool):
            return str(node.value)
        if isinstance(node, dsl_nodes.Action) or isinstance(node, dsl_nodes.IntFeature):
            return node.name
        if isinstance(node, dsl_nodes.BoolFeature):
            return node.name

        if isinstance(node, dsl_nodes.Program):
            m = self.parse_node_to_str(node.children[0])
            return f"DEF run m( {m} m)"
        if isinstance(node, dsl_nodes.While):
            c = self.parse_node_to_str(node.children[0])
            w = self.parse_node_to_str(node.children[1])
            return f"WHILE c( {c} c) w( {w} w)"
        if isinstance(node, dsl_nodes.Repeat):
            n = self.parse_node_to_str(node.children[0])
            r = self.parse_node_to_str(node.children[1])
            return f"REPEAT {n} r( {r} r)"
        if isinstance(node, dsl_nodes.If):
            c = self.parse_node_to_str(node.children[0])
            i = self.parse_node_to_str(node.children[1])
            return f"IF c( {c} c) i( {i} i)"
        if isinstance(node, dsl_nodes.ITE):
            c = self.parse_node_to_str(node.children[0])
            i = self.parse_node_to_str(node.children[1])
            e = self.parse_node_to_str(node.children[2])
            return f"IFELSE c( {c} c) i( {i} i) ELSE e( {e} e)"
        if isinstance(node, dsl_nodes.Concatenate):
            s1 = self.parse_node_to_str(node.children[0])
            s2 = self.parse_node_to_str(node.children[1])
            return f"{s1} {s2}"
        if isinstance(node, dsl_nodes.Not):
            c = self.parse_node_to_str(node.children[0])
            return f"not c( {c} c)"

        raise Exception(f"Unknown node type: {type(node)}")

    # ---------------------------------------------------------- from string
    def parse_str_list_to_node(self, prog_str_list: list[str]) -> dsl_nodes.BaseNode:
        if prog_str_list[0] in self.actions:
            if len(prog_str_list) > 1:
                s1 = dsl_nodes.Action(prog_str_list[0])
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return dsl_nodes.Action(prog_str_list[0])

        if prog_str_list[0] in self.bool_features:
            return dsl_nodes.BoolFeature(prog_str_list[0])

        if prog_str_list[0] == "<HOLE>":
            if len(prog_str_list) > 1:
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(None, s2)
            return None

        if prog_str_list[0] == "DEF":
            assert prog_str_list[1] == "run", "Invalid program"
            assert prog_str_list[2] == "m(", "Invalid program"
            assert prog_str_list[-1] == "m)", "Invalid program"
            m = self.parse_str_list_to_node(prog_str_list[3:-1])
            return dsl_nodes.Program.new(m)

        elif prog_str_list[0] == "IF":
            c_end = _find_close_token(prog_str_list, "c", 1)
            i_end = _find_close_token(prog_str_list, "i", c_end + 1)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            i = self.parse_str_list_to_node(prog_str_list[c_end + 2: i_end])
            if i_end == len(prog_str_list) - 1:
                return dsl_nodes.If.new(c, i)
            return dsl_nodes.Concatenate.new(
                dsl_nodes.If.new(c, i),
                self.parse_str_list_to_node(prog_str_list[i_end + 1:]),
            )
        elif prog_str_list[0] == "IFELSE":
            c_end = _find_close_token(prog_str_list, "c", 1)
            i_end = _find_close_token(prog_str_list, "i", c_end + 1)
            assert prog_str_list[i_end + 1] == "ELSE", "Invalid program"
            e_end = _find_close_token(prog_str_list, "e", i_end + 2)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            i = self.parse_str_list_to_node(prog_str_list[c_end + 2: i_end])
            e = self.parse_str_list_to_node(prog_str_list[i_end + 3: e_end])
            if e_end == len(prog_str_list) - 1:
                return dsl_nodes.ITE.new(c, i, e)
            return dsl_nodes.Concatenate.new(
                dsl_nodes.ITE.new(c, i, e),
                self.parse_str_list_to_node(prog_str_list[e_end + 1:]),
            )
        elif prog_str_list[0] == "WHILE":
            c_end = _find_close_token(prog_str_list, "c", 1)
            w_end = _find_close_token(prog_str_list, "w", c_end + 1)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            w = self.parse_str_list_to_node(prog_str_list[c_end + 2: w_end])
            if w_end == len(prog_str_list) - 1:
                return dsl_nodes.While.new(c, w)
            return dsl_nodes.Concatenate.new(
                dsl_nodes.While.new(c, w),
                self.parse_str_list_to_node(prog_str_list[w_end + 1:]),
            )
        elif prog_str_list[0] == "REPEAT":
            n = self.parse_str_list_to_node([prog_str_list[1]])
            r_end = _find_close_token(prog_str_list, "r", 2)
            r = self.parse_str_list_to_node(prog_str_list[3:r_end])
            if r_end == len(prog_str_list) - 1:
                return dsl_nodes.Repeat.new(n, r)
            return dsl_nodes.Concatenate.new(
                dsl_nodes.Repeat.new(n, r),
                self.parse_str_list_to_node(prog_str_list[r_end + 1:]),
            )
        elif prog_str_list[0] == "not":
            assert prog_str_list[1] == "c(", "Invalid program"
            assert prog_str_list[-1] == "c)", "Invalid program"
            c = self.parse_str_list_to_node(prog_str_list[2:-1])
            return dsl_nodes.Not.new(c)
        elif prog_str_list[0].startswith("R="):
            num = int(prog_str_list[0].replace("R=", ""))
            return dsl_nodes.ConstInt(num)
        elif prog_str_list[0] in ["True", "False"]:
            return dsl_nodes.ConstBool(prog_str_list[0] == "True")
        else:
            raise Exception(f"Unrecognized token: {prog_str_list[0]}.")
