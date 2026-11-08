"""Trusted implementation templates; candidate mode is never labeled to actors."""
import ast


def implementation_variant(source, style):
    if style == 'direct':
        return source
    tree = ast.parse(source)
    if style == 'renamed':
        mapping = {'state':'scene_input','bits':'flags','frame':'view_state','positions':'rectangles'}
        class Rename(ast.NodeTransformer):
            def visit_Name(self,node):
                node.id = mapping.get(node.id,node.id)
                return node
            def visit_arg(self,node):
                node.arg = mapping.get(node.arg,node.arg)
                return node
        return ast.unparse(Rename().visit(tree))+'\n'
    if style == 'dispatch':
        # A genuine extra function boundary, not merely renaming local variables.
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name=='solve':
                node.name='evaluate_history'
        source = ast.unparse(tree)
        return source+'\n\ndef solve(scene):\n    result = evaluate_history(scene)\n    return {key: result[key] for key in result}\n'
    raise ValueError('Unknown implementation style')
