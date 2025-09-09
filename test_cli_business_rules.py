#!/usr/bin/env python3
"""
Comprehensive CLI Integration Test for Business Rule Validation
Tests the complete CLI workflow for business rule validation.
"""

import subprocess
import json
import os
import tempfile
from pathlib import Path

def run_cli_command(cmd: list) -> tuple:
    """Run a CLI command and return result."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd="/Users/davidschaaf/projects/python_projects/sql-testing-suite"
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def test_cli_business_rules_integration():
    """Test the complete CLI business rules integration."""
    print("🧪 Testing Business Rules CLI Integration")
    print("=" * 60)
    
    # Test 1: CLI Help
    print("\n1️⃣  Testing CLI Help...")
    code, stdout, stderr = run_cli_command([
        "python", "-m", "sqltest.cli.main", "business-rules", "--help"
    ])
    
    if code == 0 and "Execute business rule validation" in stdout:
        print("   ✅ CLI help working correctly")
    else:
        print(f"   ❌ CLI help failed: {code}")
        return False
    
    # Test 2: Basic Rule Execution
    print("\n2️⃣  Testing Basic Rule Execution...")
    code, stdout, stderr = run_cli_command([
        "python", "-m", "sqltest.cli.main", 
        "--config", "test_config.yaml",
        "business-rules", 
        "--rule-set", "sample_business_rules.yaml"
    ])
    
    if code == 1 and "critical violation(s) found" in stdout:
        print("   ✅ Basic rule execution working correctly")
        print("   ✅ Correctly detected violations and returned exit code 1")
    else:
        print(f"   ❌ Basic rule execution failed: {code}")
        return False
    
    # Test 3: Tag Filtering
    print("\n3️⃣  Testing Tag Filtering...")
    code, stdout, stderr = run_cli_command([
        "python", "-m", "sqltest.cli.main", 
        "--config", "test_config.yaml",
        "business-rules", 
        "--rule-set", "sample_business_rules.yaml",
        "--tags", "completeness"
    ])
    
    if code == 0 and "All business rules passed successfully!" in stdout:
        print("   ✅ Tag filtering working correctly")
        print("   ✅ Completeness rules passed as expected")
    else:
        print(f"   ❌ Tag filtering failed: {code}")
        return False
    
    # Test 4: JSON Output Format
    print("\n4️⃣  Testing JSON Output Format...")
    code, stdout, stderr = run_cli_command([
        "python", "-m", "sqltest.cli.main", 
        "--config", "test_config.yaml",
        "business-rules", 
        "--rule-set", "sample_business_rules.yaml",
        "--tags", "completeness",
        "--format", "json"
    ])
    
    if code == 0:
        try:
            # Extract JSON from stdout - find the main JSON block
            json_start = stdout.find('{')
            if json_start != -1:
                # Find the matching closing brace by counting braces
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(stdout)):
                    if stdout[i] == '{':
                        brace_count += 1
                    elif stdout[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                json_content = stdout[json_start:json_end]
                parsed_json = json.loads(json_content)
                
                if "validation_results" in parsed_json:
                    print("   ✅ JSON output format working correctly")
                    print(f"   ✅ Parsed {len(parsed_json['validation_results'])} validation results")
                else:
                    print("   ❌ JSON output missing expected fields")
                    return False
            else:
                print("   ❌ No JSON content found in output")
                return False
        except json.JSONDecodeError as e:
            print(f"   ❌ Invalid JSON output: {e}")
            return False
    else:
        print(f"   ❌ JSON output test failed: {code}")
        return False
    
    # Test 5: Export Functionality
    print("\n5️⃣  Testing Export Functionality...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        export_file = tmp.name
    
    try:
        code, stdout, stderr = run_cli_command([
            "python", "-m", "sqltest.cli.main", 
            "--config", "test_config.yaml",
            "business-rules", 
            "--rule-set", "sample_business_rules.yaml",
            "--tags", "data_quality",
            "--output", export_file
        ])
        
        if code == 1:  # Expect violations
            if os.path.exists(export_file):
                with open(export_file, 'r') as f:
                    exported_data = json.load(f)
                
                if "summary" in exported_data and "validation_results" in exported_data:
                    print("   ✅ Export functionality working correctly")
                    print(f"   ✅ Exported file contains {len(exported_data['validation_results'])} results")
                else:
                    print("   ❌ Exported file missing expected structure")
                    return False
            else:
                print("   ❌ Export file was not created")
                return False
        else:
            print(f"   ❌ Export test failed: {code}")
            return False
    
    finally:
        if os.path.exists(export_file):
            os.unlink(export_file)
    
    # Test 6: Verbose Output
    print("\n6️⃣  Testing Verbose Output...")
    code, stdout, stderr = run_cli_command([
        "python", "-m", "sqltest.cli.main", 
        "--config", "test_config.yaml",
        "business-rules", 
        "--rule-set", "sample_business_rules.yaml",
        "--tags", "data_quality",
        "--verbose"
    ])
    
    if code == 1 and "Detailed Rule Results" in stdout:
        print("   ✅ Verbose output working correctly")
        print("   ✅ Shows detailed rule results as expected")
    else:
        print(f"   ❌ Verbose output test failed: {code}")
        return False
    
    # Test 7: Parallel vs Sequential Execution
    print("\n7️⃣  Testing Execution Modes...")
    
    # Test parallel
    code_parallel, stdout_parallel, stderr_parallel = run_cli_command([
        "python", "-m", "sqltest.cli.main", 
        "--config", "test_config.yaml",
        "business-rules", 
        "--rule-set", "sample_business_rules.yaml",
        "--tags", "data_quality",
        "--parallel"
    ])
    
    # Test sequential  
    code_sequential, stdout_sequential, stderr_sequential = run_cli_command([
        "python", "-m", "sqltest.cli.main", 
        "--config", "test_config.yaml",
        "business-rules", 
        "--rule-set", "sample_business_rules.yaml",
        "--tags", "data_quality",
        "--sequential"
    ])
    
    if code_parallel == 1 and code_sequential == 1:
        print("   ✅ Both parallel and sequential execution working")
        print("   ✅ Both modes detected violations correctly")
    else:
        print(f"   ❌ Execution modes test failed: parallel={code_parallel}, sequential={code_sequential}")
        return False
    
    # Test 8: Error Handling
    print("\n8️⃣  Testing Error Handling...")
    code, stdout, stderr = run_cli_command([
        "python", "-m", "sqltest.cli.main", 
        "--config", "test_config.yaml",
        "business-rules"
        # No rule set specified - should show error
    ])
    
    if code == 0 and "Either --rule-set or --directory must be specified" in stdout:
        print("   ✅ Error handling working correctly")
        print("   ✅ Shows helpful error message for missing arguments")
    else:
        print(f"   ❌ Error handling test failed: {code}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL CLI BUSINESS RULES INTEGRATION TESTS PASSED!")
    print("=" * 60)
    
    print("\n📊 Test Summary:")
    print("   ✅ CLI Help and Documentation")
    print("   ✅ Basic Rule Execution")
    print("   ✅ Tag-based Filtering")
    print("   ✅ JSON Output Format")
    print("   ✅ Export Functionality")
    print("   ✅ Verbose Output")
    print("   ✅ Parallel/Sequential Execution")
    print("   ✅ Error Handling")
    
    print("\n🚀 Business Rules CLI is fully integrated and working!")
    return True

if __name__ == "__main__":
    success = test_cli_business_rules_integration()
    if not success:
        exit(1)
