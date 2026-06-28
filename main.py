"""
Zero-config CLI that scans Python codebases for dangerous execution vectors (eval, exec, subprocess) to auto-generate a 

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: Unlike anthropics/defending-code-reference-harness which is a complex, multi-step orchestration harness, this provides instant, AST-based static analysis in a single file with zero setup, giving immed
"""
#!/usr/bin/env python3
"""
Sentinel Scout - Static Analysis Security Tool

A zero-config CLI scanner for identifying dangerous execution vectors in 
Python codebases. It recursively traverses directories, parses Abstract 
Syntax Trees (AST), and generates a structured Risk Manifest.

Usage Examples:
    # Scan current directory
    python3 sentinel_scout.py

    # Scan specific directory
    python3 sentinel_scout.py --path /path/to/project

    # Output to a file (redirect stdout)
    python3 sentinel_scout.py --path ./src > risk_manifest.json

    # Verbose mode (prints logs to stderr, keeps stdout as JSON)
    python3 sentinel_scout.py --verbose

Environment Variables:
    SENTINEL_API_KEY: Optional API key for external enrichment (gracefully ignored if missing).
    SENTINEL_API_URL: Optional endpoint for enrichment (defaults to a placeholder).
"""

import argparse
import ast
import json
import logging
import os
import sys
import time
import typing
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

# Attempt to import requests for the optional enrichment feature
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# -----------------------------------------------------------------------------
# Data Structures & Configuration
# -----------------------------------------------------------------------------

@dataclass
class RiskFinding:
    """Represents a single security risk finding in the codebase."""
    file_path: str
    line_number: int
    column_offset: int
    risk_type: str
    severity: str
    code_snippet: str
    function_name: Optional[str] = None
    external_ref: Optional[str] = None  # Populated if API enrichment is available

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Config:
    """Configuration holder for the scanner."""
    def __init__(self, target_path: str = ".", verbose: bool = False):
        self.target_path = Path(target_path).resolve()
        self.verbose = verbose
        self.api_key = os.getenv("SENTINEL_API_KEY")
        self.api_url = os.getenv("SENTINEL_API_URL", "https://api.example.com/enrich")
        self.max_recursion_depth = 100
        self.ignore_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox"}
        self.ignore_files = {"setup.py"} # Often contains setup logic which might look risky but is standard

    @property
    def enrichment_enabled(self) -> bool:
        return bool(self.api_key and REQUESTS_AVAILABLE)


# -----------------------------------------------------------------------------
# AST Logic & Heuristics
# -----------------------------------------------------------------------------

class DangerousCallVisitor(ast.NodeVisitor):
    """
    AST Visitor that identifies dangerous function calls and imports.
    """
    
    # Mapping of module/function names to risk categories and default severity
    RISK_VECTORS = {
        'eval': {'type': 'Dynamic Code Execution', 'severity': 'HIGH'},
        'exec': {'type': 'Dynamic Code Execution', 'severity': 'HIGH'},
        'compile': {'type': 'Dynamic Code Compilation', 'severity': 'MEDIUM'},
        '__import__': {'type': 'Bypass Import Restrictions', 'severity': 'MEDIUM'},
        'open': {'type': 'File Operation', 'severity': 'LOW'}, # Context dependent, but a vector
        'os.system': {'type': 'System Shell Command', 'severity': 'CRITICAL'},
        'subprocess.run': {'type': 'Subprocess Execution', 'severity': 'HIGH'},
        'subprocess.call': {'type': 'Subprocess Execution', 'severity': 'HIGH'},
        'subprocess.Popen': {'type': 'Subprocess Execution', 'severity': 'HIGH'},
        'subprocess.check_output': {'type': 'Subprocess Execution', 'severity': 'HIGH'},
        'pickle.loads': {'type': 'Unsafe Deserialization', 'severity': 'CRITICAL'},
        'pickle.load': {'type': 'Unsafe Deserialization', 'severity': 'CRITICAL'},
        'marshal.loads': {'type': 'Unsafe Deserialization', 'severity': 'CRITICAL'},
        'shelve.open': {'type': 'Unsafe Deserialization', 'severity': 'MEDIUM'},
        'yaml.load': {'type': 'Unsafe Deserialization', 'severity': 'MEDIUM'},
    }

    def __init__(self, file_path: str, source_lines: List[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.findings: List[RiskFinding] = []
        self.imports: Dict[str, str] = {} # Maps alias to original module name

    def _get_code_snippet(self, node: ast.AST) -> str:
        """Extracts the line of code where the node is found."""
        start_line = node.lineno - 1
        if 0 <= start_line < len(self.source_lines):
            return self.source_lines[start_line].strip()
        return ""

    def _resolve_name(self, node: ast.AST) -> Optional[str]:
        """
        Resolve the dotted name of a Call node.
        Examples: 
            Name(id='eval') -> 'eval'
            Call(func=Attribute(value=Name(id='os'), attr='system')) -> 'os.system'
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._resolve_name(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._resolve_name(node.func)
        return None

    def _check_risk(self, node: ast.AST, call_name: str):
        """Check if the call name is in our risk vectors and record finding."""
        # Check direct match or module.submodule match
        risk_info = self.RISK_VECTORS.get(call_name)
        
        # Fallback for common patterns like 'subprocess' usage 
        # (e.g. if someone does `from subprocess import Popen`)
        if not risk_info:
            # Check if it matches an imported function that is dangerous
            # e.g., `loads` might be imported from `pickle`
            if call_name in self.imports:
                full_path = f"{self.imports[call_name]}.{call_name}"
                risk_info = self.RISK_VECTORS.get(full_path)

        if risk_info:
            # Basic logic to filter out comments (e.g., # nosec) could go here
            # For now, we report strictly.
            finding = RiskFinding(
                file_path=self.file_path,
                line_number=node.lineno,
                column_offset=node.col_offset if hasattr(node, 'col_offset') else 0,
                risk_type=risk_info['type'],
                severity=risk_info['severity'],
                code_snippet=self._get_code_snippet(node),
                function_name=call_name
            )
            self.findings.append(finding)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module
        if module:
            for alias in node.names:
                imported_name = alias.asname or alias.name
                # Store mapping: 'load' -> 'pickle'
                self.imports[imported_name] = module
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        call_name = self._resolve_name(node)
        if call_name:
            self._check_risk(node, call_name)
        self.generic_visit(node)


# -----------------------------------------------------------------------------
# Core Scanner Logic
# -----------------------------------------------------------------------------

class SecurityScanner:
    """Orchestrates the file system walk and analysis."""

    def __init__(self, config: Config):
        self.config = config
        self.findings: List[RiskFinding] = []
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger("SentinelScout")
        self.logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        handler = logging.StreamHandler(sys.stderr) # Log to stderr to keep stdout clean for JSON
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _is_excluded(self, path: Path) -> bool:
        """Check if path should be ignored based on config."""
        # Check directory exclusions
        for part in path.parts:
            if part in self.config.ignore_dirs:
                return True
        # Check file exclusions
        if path.name in self.config.ignore_files:
            return True
        return False

    def scan_file(self, file_path: Path) -> Optional[List[RiskFinding]]:
        """Parses a single python file and returns findings."""
        try:
            self.logger.debug(f"Scanning: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parse the AST. Handle syntax errors gracefully.
            try:
                tree = ast.parse(source, filename=str(file_path))
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in {file_path}: {e.msg}")
                return None

            source_lines = source.splitlines()
            visitor = DangerousCallVisitor(str(file_path), source_lines)
            visitor.visit(tree)
            return visitor.findings

        except IOError as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            return None

    def run_scan(self) -> List[RiskFinding]:
        """Walks the directory tree and aggregates findings."""
        if not self.config.target_path.exists():
            self.logger.critical(f"Target path does not exist: {self.config.target_path}")
            sys.exit(1)

        self.logger.info(f"Starting scan on: {self.config.target_path}")
        
        py_files = list(self.config.target_path.rglob("*.py"))
        total_files = len(py_files)
        processed_count = 0

        for py_file in py_files:
            if self._is_excluded(py_file):
                continue
            
            findings = self.scan_file(py_file)
            if findings:
                self.findings.extend(findings)
            
            processed_count += 1
            if self.config.verbose and processed_count % 50 == 0:
                self.logger.debug(f"Progress: {processed_count}/{total_files} files scanned.")

        self.logger.info(f"Scan complete. Total findings: {len(self.findings)}")
        return self.findings

# -----------------------------------------------------------------------------
# External Enrichment (Graceful Degradation)
# -----------------------------------------------------------------------------

class EnrichmentService:
    """Handles optional external API lookups for findings."""

    @staticmethod
    def enrich(config: Config, findings: List[RiskFinding]) -> List[RiskFinding]:
        if not config.enrichment_enabled or not findings:
            return findings

        logging.info("Enrichment available. Attempting to lookup external references...")
        
        enriched_findings = []
        # Batch processing logic could go here, but we'll do a simple loop for robustness
        headers = {"Authorization": f"Bearer {config.api_key}"}
        
        for finding in findings:
            # Create a payload
            payload = {
                "type": finding.risk_type,
                "signature": finding.function_name
            }
            
            try:
                # NOTE: This will likely fail 404/500 since the URL is a placeholder.
                # This demonstrates the "graceful degradation" requirement.
                response = requests.post(config.api_url, json=payload, headers=headers, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    finding.external_ref = data.get("cve_id", "UNKNOWN_REF")
                else:
                    finding.external_ref = "NO_EXT_REF"
            except Exception as e:
                # If network fails or API is down, we continue with local data
                logging.debug(f"Enrichment failed for {finding.function_name}: {e}")
                finding.external_ref = None # Erase if failed
            
            enriched_findings.append(finding)
            
        return enriched_findings

# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------

def parse_arguments() -> Config:
    """Sets up argument parsing."""
    parser = argparse.ArgumentParser(
        description="Sentinel Scout: Python codebase security risk scanner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Output is a JSON array of risk objects sent to stdout."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Path to the root directory to scan. Defaults to current directory."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging to stderr."
    )
    args = parser.parse_args()
    
    return Config(target_path=args.path, verbose=args.verbose)

def main():
    # 1. Setup
    config = parse_arguments()
    scanner = SecurityScanner(config)

    # 2. Scan
    findings = scanner.run_scan()

    # 3. Enrich (Optional)
    if REQUESTS_AVAILABLE:
        findings = EnrichmentService.enrich(config, findings)

    # 4. Output JSON
    # Convert dataclass objects to dicts for JSON serialization
    output_data = [f.to_dict() for f in findings]
    
    # Print metadata and results
    manifest = {
        "scan_timestamp": time.time(),
        "target_path": str(config.target_path),
        "total_risks_found": len(output_data),
        "findings": output_data
    }

    try:
        json.dump(manifest, sys.stdout, indent=2)
    except IOError:
        # Handle broken pipe (e.g., user piped to head and it closed)
        pass
    finally:
        sys.stdout.write("\n")

if __name__ == "__main__":
    main()