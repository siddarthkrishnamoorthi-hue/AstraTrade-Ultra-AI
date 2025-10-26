"""
Code quality and optimization utilities
"""
import ast
import inspect
import sys
from pathlib import Path
from typing import Set, List, Dict
import logging

class CodeQualityChecker:
    def __init__(self):
        self.logger = logging.getLogger("CodeQuality")
        self.unused_imports = set()
        self.unused_variables = set()
        self.complexity_issues = []
        
    def analyze_project(self, project_path: str):
        """Analyze entire project for code quality issues"""
        project_dir = Path(project_path)
        
        for python_file in project_dir.rglob("*.py"):
            self.analyze_file(python_file)
            
    def analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            
            # Report issues
            if analyzer.unused_imports:
                self.logger.warning(
                    f"Unused imports in {file_path}: {analyzer.unused_imports}"
                )
                
            if analyzer.unused_variables:
                self.logger.warning(
                    f"Unused variables in {file_path}: {analyzer.unused_variables}"
                )
                
            if analyzer.complexity_issues:
                self.logger.warning(
                    f"Complexity issues in {file_path}: {analyzer.complexity_issues}"
                )
                
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}")
            
class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.unused_imports = set()
        self.unused_variables = set()
        self.complexity_issues = []
        self.used_names = set()
        self.imports = {}
        self.scope_variables = []
        
    def visit_Import(self, node):
        """Track imports"""
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name
            
    def visit_ImportFrom(self, node):
        """Track from imports"""
        for alias in node.names:
            self.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            
    def visit_Name(self, node):
        """Track name usage"""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
            
    def visit_FunctionDef(self, node):
        """Analyze function complexity"""
        # Check cyclomatic complexity
        complexity = self._calculate_complexity(node)
        if complexity > 10:  # McCabe complexity threshold
            self.complexity_issues.append({
                'function': node.name,
                'complexity': complexity,
                'line': node.lineno
            })
            
        # Check function length
        source = ast.get_source_segment(self.source, node)
        if source and len(source.splitlines()) > 50:  # Max lines threshold
            self.complexity_issues.append({
                'function': node.name,
                'issue': 'too_long',
                'line': node.lineno
            })
            
        self.generic_visit(node)
        
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                               ast.ExceptHandler, ast.AsyncWith,
                               ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
        
class CodeOptimizer:
    def __init__(self):
        self.logger = logging.getLogger("CodeOptimizer")
        
    def optimize_imports(self, file_path: Path):
        """Remove unused imports"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            
            # Remove unused imports
            unused = set(analyzer.imports.keys()) - analyzer.used_names
            if unused:
                self.logger.info(f"Removing unused imports: {unused}")
                # Implement import removal logic
                
        except Exception as e:
            self.logger.error(f"Error optimizing imports: {str(e)}")
            
    def remove_dead_code(self, file_path: Path):
        """Remove unreachable and dead code"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            tree = ast.parse(code)
            transformer = DeadCodeTransformer()
            transformed = transformer.visit(tree)
            
            # Write optimized code
            optimized_code = ast.unparse(transformed)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(optimized_code)
                
        except Exception as e:
            self.logger.error(f"Error removing dead code: {str(e)}")
            
class DeadCodeTransformer(ast.NodeTransformer):
    """Transform AST to remove dead code"""
    
    def visit_If(self, node):
        """Remove if statements with constant conditions"""
        self.generic_visit(node)
        
        try:
            test_value = ast.literal_eval(node.test)
            if test_value:
                return node.body
            else:
                return node.orelse
        except:
            return node
            
    def visit_While(self, node):
        """Remove while loops with constant False conditions"""
        self.generic_visit(node)
        
        try:
            test_value = ast.literal_eval(node.test)
            if test_value is False:
                return None
        except:
            pass
            
        return node
        
def optimize_project(project_path: str):
    """Run all code optimizations on project"""
    checker = CodeQualityChecker()
    optimizer = CodeOptimizer()
    
    # Analyze code quality
    checker.analyze_project(project_path)
    
    # Optimize each Python file
    project_dir = Path(project_path)
    for python_file in project_dir.rglob("*.py"):
        optimizer.optimize_imports(python_file)
        optimizer.remove_dead_code(python_file)
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        optimize_project(sys.argv[1])
    else:
        print("Please provide project path")