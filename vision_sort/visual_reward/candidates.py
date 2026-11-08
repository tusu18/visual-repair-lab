"""Public proposed-patch templates for the separate audit curriculum.

Candidate constants are visible engineering code; their correctness is unknown.
No reference selector, private target identifier or grader source is exported.
"""
import ast
import inspect

from ..visual_causal.tasks import FUNCTIONS, oracle_source


def candidate_source(family, mask):
    bits = [(mask >> i) & 1 for i in range(6)]
    names = {f.__name__ for f in FUNCTIONS[family] if 'bits' in inspect.signature(f).parameters}

    class Specialize(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'bits':
                if isinstance(node.slice, ast.Constant):
                    return ast.copy_location(ast.Constant(bits[node.slice.value]), node)
                return ast.copy_location(ast.Subscript(
                    value=ast.List(elts=[ast.Constant(v) for v in bits], ctx=ast.Load()),
                    slice=self.visit(node.slice), ctx=ast.Load()), node)
            node = self.generic_visit(node)
            if isinstance(node.value, (ast.List, ast.Tuple)) and isinstance(node.slice, ast.Constant):
                return node.value.elts[node.slice.value]
            return node

        def visit_UnaryOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
                return ast.copy_location(ast.Constant(not node.operand.value), node)
            return node

        def visit_If(self, node):
            node = self.generic_visit(node)
            return (node.body if node.test.value else node.orelse) if isinstance(node.test, ast.Constant) else node

        def visit_IfExp(self, node):
            node = self.generic_visit(node)
            return (node.body if node.test.value else node.orelse) if isinstance(node.test, ast.Constant) else node

        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            node.args.args = [a for a in node.args.args if a.arg != 'bits']
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in names:
                node.args = node.args[:1]
            return node

    tree = Specialize().visit(ast.parse(oracle_source(family, 0)))
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
