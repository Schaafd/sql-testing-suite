#!/usr/bin/env python3
"""
Test script for Business Rules YAML configuration loading
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqltest.modules.business_rules.config_loader import BusinessRuleConfigLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_yaml_loading():
    """Test loading business rules from YAML configuration."""
    
    print("🔧 SQLTest Pro Business Rules YAML Configuration Test")
    print("=" * 65)
    
    # Initialize config loader
    config_loader = BusinessRuleConfigLoader()
    
    # Test loading from the sample YAML file
    yaml_file = "sample_business_rules.yaml"
    
    try:
        print(f"\n📄 Loading configuration from: {yaml_file}")
        
        # Load rule set from file
        rule_set = config_loader.load_rule_set_from_file(yaml_file)
        
        print(f"\n✅ Successfully loaded rule set!")
        print(f"  Name: {rule_set.name}")
        print(f"  Description: {rule_set.description}")
        print(f"  Total Rules: {len(rule_set.rules)}")
        print(f"  Enabled: {rule_set.enabled}")
        print(f"  Parallel Execution: {rule_set.parallel_execution}")
        print(f"  Max Concurrent Rules: {rule_set.max_concurrent_rules}")
        print(f"  Tags: {', '.join(rule_set.tags)}")
        
        print(f"\n📋 Rule Details:")
        print("-" * 50)
        
        for i, rule in enumerate(rule_set.rules, 1):
            print(f"\n{i}. {rule.name}")
            print(f"   Type: {rule.rule_type.value}")
            print(f"   Severity: {rule.severity.value}")
            print(f"   Scope: {rule.scope.value}")
            print(f"   Enabled: {rule.enabled}")
            print(f"   Tags: {', '.join(rule.tags)}")
            print(f"   Timeout: {rule.timeout_seconds}s")
            
            if rule.max_violation_count is not None:
                print(f"   Max Violations: {rule.max_violation_count}")
            
            if rule.dependencies:
                print(f"   Dependencies: {', '.join(rule.dependencies)}")
            
            # Show first few lines of SQL query
            if rule.sql_query:
                query_lines = rule.sql_query.strip().split('\n')
                first_line = query_lines[0].strip()
                print(f"   Query: {first_line}{'...' if len(query_lines) > 1 else ''}")
        
        # Test rule set validation
        print(f"\n🔍 Rule Set Validation:")
        enabled_rules = rule_set.get_enabled_rules()
        disabled_rules = [r for r in rule_set.rules if not r.enabled]
        
        print(f"  ✅ Enabled Rules: {len(enabled_rules)}")
        print(f"  ⏸️  Disabled Rules: {len(disabled_rules)}")
        
        # Group rules by type
        rule_types = {}
        for rule in rule_set.rules:
            rule_type = rule.rule_type.value
            if rule_type not in rule_types:
                rule_types[rule_type] = 0
            rule_types[rule_type] += 1
        
        print(f"\n📊 Rules by Type:")
        for rule_type, count in rule_types.items():
            print(f"  {rule_type}: {count}")
        
        # Group rules by severity
        rule_severities = {}
        for rule in rule_set.rules:
            severity = rule.severity.value
            if severity not in rule_severities:
                rule_severities[severity] = 0
            rule_severities[severity] += 1
        
        print(f"\n⚠️  Rules by Severity:")
        for severity, count in rule_severities.items():
            print(f"  {severity}: {count}")
        
        # Test dependency analysis
        dependencies = set()
        for rule in rule_set.rules:
            dependencies.update(rule.dependencies)
        
        print(f"\n🔗 Dependency Analysis:")
        if dependencies:
            print(f"  Dependencies found: {', '.join(dependencies)}")
            
            # Check if all dependencies are satisfied
            rule_names = {rule.name for rule in rule_set.rules}
            missing_deps = dependencies - rule_names
            if missing_deps:
                print(f"  ⚠️  Missing dependencies: {', '.join(missing_deps)}")
            else:
                print(f"  ✅ All dependencies satisfied")
        else:
            print(f"  No dependencies defined")
        
        print(f"\n✨ YAML configuration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error loading YAML configuration: {str(e)}")
        logger.exception("YAML loading failed")

if __name__ == "__main__":
    test_yaml_loading()
