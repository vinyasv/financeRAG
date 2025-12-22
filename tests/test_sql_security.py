#!/usr/bin/env python3
"""Test SQL injection protection measures."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.sqlite_store import (
    validate_sql_query, 
    add_limit_clause, 
    SecurityError,
    SQLiteStore
)


def test_validate_sql_query():
    """Test SQL validation function."""
    print("=" * 60)
    print("Testing SQL Validation")
    print("=" * 60)
    
    # Valid queries that should pass
    valid_queries = [
        "SELECT * FROM documents",
        "SELECT id, name FROM users WHERE id = 1",
        "SELECT revenue FROM berkshire_annual_performance WHERE year = '2024'",
        "select * from table1",  # lowercase
        "SELECT a.id, b.name FROM table1 a JOIN table2 b ON a.id = b.id",
        "SELECT created_at, updated_at FROM logs",  # columns with 'at' suffix
    ]
    
    print("\n✅ Testing VALID queries (should pass):")
    all_passed = True
    for query in valid_queries:
        is_valid, error = validate_sql_query(query)
        status = "✓" if is_valid else f"✗ ({error})"
        print(f"  {status}: {query[:60]}...")
        if not is_valid:
            all_passed = False
    
    # Invalid/malicious queries that should be blocked
    invalid_queries = [
        ("DROP TABLE documents", "DROP"),
        ("DELETE FROM documents WHERE 1=1", "DELETE"),
        ("INSERT INTO documents VALUES (1)", "INSERT"),
        ("UPDATE documents SET name='hack'", "UPDATE"),
        ("SELECT * FROM users; DROP TABLE users;--", "multiple statements"),
        ("SELECT * FROM users; DELETE FROM users", "multiple statements"),
        ("ALTER TABLE users ADD COLUMN hack TEXT", "ALTER"),
        ("CREATE TABLE hack (id INT)", "CREATE"),
        ("TRUNCATE TABLE users", "TRUNCATE"),
        ("SELECT * FROM users -- comment injection", "comments"),
        ("SELECT * FROM users /* inline comment */", "comments"),
        ("PRAGMA table_info(users)", "PRAGMA"),
        ("ATTACH DATABASE 'hack.db' AS hack", "ATTACH"),
        ("", "empty"),
    ]
    
    print("\n🚫 Testing INVALID queries (should be blocked):")
    for query, expected_reason in invalid_queries:
        is_valid, error = validate_sql_query(query)
        status = "✓ BLOCKED" if not is_valid else "✗ ALLOWED (BAD!)"
        print(f"  {status}: {query[:50]}... [{expected_reason}]")
        if is_valid:
            all_passed = False
            print(f"    ⚠️  WARNING: This query should have been blocked!")
    
    return all_passed


def test_add_limit_clause():
    """Test automatic LIMIT clause addition."""
    print("\n" + "=" * 60)
    print("Testing LIMIT Clause Addition")
    print("=" * 60)
    
    test_cases = [
        ("SELECT * FROM users", "SELECT * FROM users LIMIT 10000"),
        ("SELECT * FROM users LIMIT 100", "SELECT * FROM users LIMIT 100"),  # unchanged
        ("SELECT * FROM users;", "SELECT * FROM users LIMIT 10000"),  # removes semicolon
        ("select * from t", "select * from t LIMIT 10000"),
    ]
    
    all_passed = True
    for input_sql, expected in test_cases:
        result = add_limit_clause(input_sql)
        status = "✓" if result == expected else "✗"
        print(f"  {status}: {input_sql}")
        print(f"     → {result}")
        if result != expected:
            print(f"     Expected: {expected}")
            all_passed = False
    
    return all_passed


def test_execute_query_security():
    """Test that execute_query properly validates SQL."""
    print("\n" + "=" * 60)
    print("Testing execute_query() Security")
    print("=" * 60)
    
    store = SQLiteStore()
    all_passed = True
    
    # Test that malicious queries raise SecurityError
    malicious_queries = [
        "DROP TABLE documents",
        "DELETE FROM documents",
        "SELECT * FROM users; DROP TABLE users",
    ]
    
    print("\n🚫 Testing malicious query blocking:")
    for query in malicious_queries:
        try:
            store.execute_query(query)
            print(f"  ✗ FAILED: {query[:40]}... (should have raised SecurityError)")
            all_passed = False
        except SecurityError as e:
            print(f"  ✓ BLOCKED: {query[:40]}... → SecurityError: {e}")
        except Exception as e:
            print(f"  ✗ WRONG ERROR: {query[:40]}... → {type(e).__name__}: {e}")
            all_passed = False
    
    # Test that valid queries work (may fail if tables don't exist, that's OK)
    print("\n✅ Testing valid query execution:")
    try:
        result = store.execute_query("SELECT 1 as test")
        print(f"  ✓ Valid query executed: {result}")
    except SecurityError as e:
        print(f"  ✗ Valid query blocked: {e}")
        all_passed = False
    except Exception as e:
        # Other errors (like no table) are OK - we're testing security
        print(f"  ✓ Query allowed (execution error is OK): {e}")
    
    return all_passed


def main():
    """Run all security tests."""
    print("\n" + "=" * 60)
    print("SQL INJECTION PROTECTION TESTS")
    print("=" * 60)
    
    results = []
    
    results.append(("SQL Validation", test_validate_sql_query()))
    results.append(("LIMIT Clause", test_add_limit_clause()))
    results.append(("Execute Query Security", test_execute_query_security()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All SQL injection protection tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
