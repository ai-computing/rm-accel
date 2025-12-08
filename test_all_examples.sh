#!/bin/bash
# Test all examples and report results

echo "================================================================================"
echo "TESTING ALL EXAMPLES"
echo "================================================================================"
echo ""

cd /home/user/rm_abstract_layer

total=0
passed=0
failed=0

for example in examples/0*.py; do
    total=$((total + 1))
    name=$(basename "$example")

    echo "Testing: $name"

    # Run example with timeout, suppress output
    # Use env to pass environment variables to timeout
    # 120s timeout for examples that test multiple engines (torchserve has 60s timeout)
    if timeout 120s env PYTHONPATH=./src .venv/bin/python "$example" >/dev/null 2>&1; then
        echo "  ✅ PASS"
        passed=$((passed + 1))
    else
        echo "  ❌ FAIL (or timeout)"
        failed=$((failed + 1))
    fi
    echo ""
done

echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Total:  $total"
echo "Passed: $passed"
echo "Failed: $failed"
echo ""

if [ $failed -eq 0 ]; then
    echo "✅ ALL TESTS PASSED!"
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    exit 1
fi
