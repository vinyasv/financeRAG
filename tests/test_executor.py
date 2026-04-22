"""Tests for the DAG executor."""

import asyncio

import pytest

from src.agent.executor import DAGExecutor, ExecutionMonitor
from src.models import ExecutionPlan, ToolCall, ToolName


class TestDAGExecutor:
    """Test the DAG executor."""
    
    @pytest.fixture
    def executor(self):
        return DAGExecutor()
    
    def test_single_step_execution(self, executor):
        """Test executing a plan with a single step."""
        plan = ExecutionPlan(
            query="Test query",
            reasoning="Test",
            steps=[
                ToolCall(
                    id="step_1",
                    tool=ToolName.CALCULATOR,
                    input="2 + 2",
                    depends_on=[],
                    description="Simple addition"
                )
            ]
        )
        
        results = asyncio.run(executor.execute(plan))
        
        assert "step_1" in results
        assert results["step_1"].success
        assert results["step_1"].result.result == 4.0  # CalculationTranscript.result
    
    def test_parallel_execution(self, executor):
        """Test that independent steps can run in parallel."""
        plan = ExecutionPlan(
            query="Test parallel",
            reasoning="Test",
            steps=[
                ToolCall(
                    id="step_1",
                    tool=ToolName.CALCULATOR,
                    input="10 + 10",
                    depends_on=[],
                    description="First calc"
                ),
                ToolCall(
                    id="step_2",
                    tool=ToolName.CALCULATOR,
                    input="20 + 20",
                    depends_on=[],
                    description="Second calc"
                )
            ]
        )
        
        results = asyncio.run(executor.execute(plan))
        
        assert results["step_1"].success
        assert results["step_1"].result.result == 20.0  # CalculationTranscript.result
        assert results["step_2"].success
        assert results["step_2"].result.result == 40.0  # CalculationTranscript.result
    
    def test_dependent_execution(self, executor):
        """Test that dependencies are resolved correctly."""
        plan = ExecutionPlan(
            query="Test dependencies",
            reasoning="Test",
            steps=[
                ToolCall(
                    id="step_1",
                    tool=ToolName.CALCULATOR,
                    input="100",
                    depends_on=[],
                    description="Get base value"
                ),
                ToolCall(
                    id="step_2",
                    tool=ToolName.CALCULATOR,
                    input="{step_1} * 2",
                    depends_on=["step_1"],
                    description="Double it"
                )
            ]
        )
        
        results = asyncio.run(executor.execute(plan))
        
        assert results["step_1"].result.result == 100.0  # CalculationTranscript.result
        assert results["step_2"].result.result == 200.0  # CalculationTranscript.result
    
    def test_complex_dag(self, executor):
        """Test a more complex DAG with multiple dependencies."""
        plan = ExecutionPlan(
            query="Complex calculation",
            reasoning="Test",
            steps=[
                ToolCall(
                    id="step_1",
                    tool=ToolName.CALCULATOR,
                    input="1000",
                    depends_on=[],
                    description="Revenue"
                ),
                ToolCall(
                    id="step_2",
                    tool=ToolName.CALCULATOR,
                    input="400",
                    depends_on=[],
                    description="Costs"
                ),
                ToolCall(
                    id="step_3",
                    tool=ToolName.CALCULATOR,
                    input="{step_1} - {step_2}",
                    depends_on=["step_1", "step_2"],
                    description="Profit"
                ),
                ToolCall(
                    id="step_4",
                    tool=ToolName.CALCULATOR,
                    input="{step_3} / {step_1} * 100",
                    depends_on=["step_1", "step_3"],
                    description="Profit margin"
                )
            ]
        )
        
        results = asyncio.run(executor.execute(plan))
        
        assert results["step_1"].result.result == 1000.0  # CalculationTranscript.result
        assert results["step_2"].result.result == 400.0  # CalculationTranscript.result
        assert results["step_3"].result.result == 600.0  # CalculationTranscript.result
        assert results["step_4"].result.result == 60.0  # 60% profit margin

    def test_execution_monitor_matches_executor_results(self, executor):
        """Monitored execution should return the same results as plain execution."""
        plan = ExecutionPlan(
            query="Monitored calculation",
            reasoning="Parity test",
            steps=[
                ToolCall(id="step_1", tool=ToolName.CALCULATOR, input="25", depends_on=[], description="Base"),
                ToolCall(id="step_2", tool=ToolName.CALCULATOR, input="{step_1} * 4", depends_on=["step_1"], description="Multiply"),
            ],
        )

        plain_results = asyncio.run(executor.execute(plan))

        monitor = ExecutionMonitor()
        monitored_results, timing = asyncio.run(monitor.execute_with_monitoring(executor, plan))

        assert plain_results.keys() == monitored_results.keys()
        for step_id in plain_results:
            assert plain_results[step_id].success == monitored_results[step_id].success
            assert plain_results[step_id].tool == monitored_results[step_id].tool
            assert plain_results[step_id].error == monitored_results[step_id].error
            assert plain_results[step_id].result.result == monitored_results[step_id].result.result

        assert timing["step_count"] == 2
        assert timing["layer_count"] == 2
        assert set(timing["step_times_ms"].keys()) == {"step_1", "step_2"}


class TestExecutionLayers:
    """Test the execution layer grouping."""
    
    def test_all_independent(self):
        """All independent steps should be in one layer."""
        plan = ExecutionPlan(
            query="Test",
            reasoning="Test",
            steps=[
                ToolCall(id="a", tool=ToolName.CALCULATOR, input="1", depends_on=[], description=""),
                ToolCall(id="b", tool=ToolName.CALCULATOR, input="2", depends_on=[], description=""),
                ToolCall(id="c", tool=ToolName.CALCULATOR, input="3", depends_on=[], description=""),
            ]
        )
        
        layers = plan.get_execution_layers()
        
        assert len(layers) == 1
        assert len(layers[0]) == 3
    
    def test_sequential_dependencies(self):
        """Sequential dependencies should create multiple layers."""
        plan = ExecutionPlan(
            query="Test",
            reasoning="Test",
            steps=[
                ToolCall(id="a", tool=ToolName.CALCULATOR, input="1", depends_on=[], description=""),
                ToolCall(id="b", tool=ToolName.CALCULATOR, input="2", depends_on=["a"], description=""),
                ToolCall(id="c", tool=ToolName.CALCULATOR, input="3", depends_on=["b"], description=""),
            ]
        )
        
        layers = plan.get_execution_layers()
        
        assert len(layers) == 3
        assert layers[0][0].id == "a"
        assert layers[1][0].id == "b"
        assert layers[2][0].id == "c"
    
    def test_diamond_dependency(self):
        """Diamond pattern: A -> B, A -> C, B+C -> D."""
        plan = ExecutionPlan(
            query="Test",
            reasoning="Test",
            steps=[
                ToolCall(id="a", tool=ToolName.CALCULATOR, input="1", depends_on=[], description=""),
                ToolCall(id="b", tool=ToolName.CALCULATOR, input="2", depends_on=["a"], description=""),
                ToolCall(id="c", tool=ToolName.CALCULATOR, input="3", depends_on=["a"], description=""),
                ToolCall(id="d", tool=ToolName.CALCULATOR, input="4", depends_on=["b", "c"], description=""),
            ]
        )
        
        layers = plan.get_execution_layers()
        
        assert len(layers) == 3
        assert layers[0][0].id == "a"
        assert set(s.id for s in layers[1]) == {"b", "c"}
        assert layers[2][0].id == "d"
