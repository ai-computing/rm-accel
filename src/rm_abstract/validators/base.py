"""
Validator Base Classes

Common data structures for system validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from enum import Enum


class TestStatus(Enum):
    """Test result status"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


@dataclass
class TestResult:
    """Single test result"""
    name: str
    status: TestStatus
    message: str = ""
    duration_ms: float = 0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status_icons = {
            TestStatus.PASS: "✓",
            TestStatus.FAIL: "✗",
            TestStatus.SKIP: "⊘",
            TestStatus.WARN: "⚠",
        }
        icon = status_icons.get(self.status, "?")
        return f"{icon} {self.name}: {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report"""
    timestamp: str = ""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    warnings: int = 0
    results: List[TestResult] = field(default_factory=list)
    
    def add_result(self, result: TestResult) -> None:
        """Add a test result to the report"""
        self.results.append(result)
        self.total_tests += 1
        
        if result.status == TestStatus.PASS:
            self.passed += 1
        elif result.status == TestStatus.FAIL:
            self.failed += 1
        elif result.status == TestStatus.SKIP:
            self.skipped += 1
        elif result.status == TestStatus.WARN:
            self.warnings += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100
    
    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"Total: {self.total_tests} | "
            f"Passed: {self.passed} | "
            f"Failed: {self.failed} | "
            f"Skipped: {self.skipped} | "
            f"Warnings: {self.warnings}"
        )

