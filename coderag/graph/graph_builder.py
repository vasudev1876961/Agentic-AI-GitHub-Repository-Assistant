import os, re, pickle
import networkx as nx
from coderag.config import REPOS_DIR, WATCHED_DIR

PY_EXT = (".py",)

def _extract_imports(code: str):
    # simple extraction; improve later with AST if needed
    imports = re.findall(r'^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))', code, flags=re.MULTILINE)
    # flatten tuple results
    cleaned = []
    for a,b in imports:
        pkg = a or b
        if pkg:
            cleaned.append(pkg.split('.')[0])
    return cleaned

def _extract_funcs(code: str):
    return re.findall(r'^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', code, flags=re.MULTILINE)

def build_code_graph(repo_dir: str) -> nx.DiGraph:
    G = nx.DiGraph()
    for root, _, files in os.walk(repo_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            node_file = f"file:{os.path.relpath(path, repo_dir)}"
            G.add_node(node_file, type="file", path=path)

            try:
                with open(path, "r", encoding="utf-8") as fh:
                    code = fh.read()
            except Exception:
                continue

            # functions
            for fn in _extract_funcs(code):
                fn_node = f"func:{os.path.relpath(path, repo_dir)}:{fn}"
                G.add_node(fn_node, type="function", file=node_file, name=fn)
                G.add_edge(node_file, fn_node, relation="defines")

            # imports
            for pkg in _extract_imports(code):
                pkg_node = f"module:{pkg}"
                G.add_node(pkg_node, type="module", name=pkg)
                G.add_edge(node_file, pkg_node, relation="imports")

    return G

def save_graph(G: nx.DiGraph, dest: str):
    with open(dest, "wb") as f:
        pickle.dump(G, f)

def load_graph(src: str) -> nx.DiGraph:
    with open(src, "rb") as f:
        return pickle.load(f)
