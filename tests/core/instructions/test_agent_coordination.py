"""Integration tests for agent coordination instructions."""

import pytest
import asyncio
from typing import Dict, Any
from agentflow.core.instructions.agent_coordination import (
    AgentSynchronizationInstruction,
    ResourceCoordinationInstruction,
    WorkflowCoordinationInstruction,
    CoordinationConfig
)
from agentflow.core.instructions.base import ExecutionContext

@pytest.fixture
def coordination_config():
    """Create test coordination configuration."""
    return CoordinationConfig(
        max_participants=3,
        timeout_seconds=5.0,
        min_success_rate=0.8,
        resource_threshold=0.9,
        coordination_mode="synchronous"
    )

@pytest.fixture
def sync_instruction(coordination_config):
    """Create test synchronization instruction."""
    return AgentSynchronizationInstruction(
        name="test_sync",
        coordination_config=coordination_config
    )

@pytest.fixture
def resource_instruction(coordination_config):
    """Create test resource coordination instruction."""
    return ResourceCoordinationInstruction(
        name="test_resource",
        coordination_config=coordination_config,
        resource_limits={
            "cpu": 100.0,
            "memory": 1024.0,
            "disk": 10000.0
        }
    )

@pytest.fixture
def workflow_instruction(coordination_config):
    """Create test workflow coordination instruction."""
    return WorkflowCoordinationInstruction(
        name="test_workflow",
        coordination_config=coordination_config,
        workflow_schema={
            "states": ["initialized", "running", "completed"],
            "actions": ["start", "update", "complete"],
            "data_schema": {
                "type": "object",
                "properties": {
                    "progress": {"type": "number"},
                    "status": {"type": "string"}
                }
            }
        }
    )

@pytest.mark.asyncio
async def test_synchronization(sync_instruction):
    """Test agent synchronization."""
    # Create test agents
    agents = ["agent1", "agent2", "agent3"]
    sync_id = "test_sync_1"
    
    # Create tasks for each agent
    async def agent_task(agent_id: str):
        context = ExecutionContext(
            params={
                "sync_id": sync_id,
                "agent_id": agent_id,
                "sync_data": {"status": "ready"}
            },
            state={},
            metadata={"timestamp": 0.0}
        )
        return await sync_instruction.execute(context)
    
    # Execute synchronization
    tasks = [agent_task(agent_id) for agent_id in agents]
    results = await asyncio.gather(*tasks)
    
    # Verify results
    for result in results:
        assert result.success
        assert "sync_data" in result.output
        assert len(result.output["sync_data"]) == len(agents)
    
    # Verify metrics
    assert sync_instruction.metrics.coordination_attempts == len(agents)
    assert sync_instruction.metrics.successful_coordinations == len(agents)
    assert sync_instruction.metrics.failed_coordinations == 0

@pytest.mark.asyncio
async def test_synchronization_timeout(sync_instruction):
    """Test synchronization timeout."""
    context = ExecutionContext(
        params={
            "sync_id": "test_sync_2",
            "agent_id": "agent1",
            "sync_data": {"status": "ready"}
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    
    # Execute with single agent (should timeout)
    result = await sync_instruction.execute(context)
    
    assert not result.success
    assert any("timeout" in str(error["error"]).lower() for error in result.errors)
    assert sync_instruction.metrics.failed_coordinations > 0

@pytest.mark.asyncio
async def test_resource_coordination(resource_instruction):
    """Test resource coordination."""
    # Create test requests
    requests = [
        {
            "agent_id": "agent1",
            "request": {"cpu": 30.0, "memory": 256.0, "disk": 1000.0}
        },
        {
            "agent_id": "agent2",
            "request": {"cpu": 40.0, "memory": 512.0, "disk": 2000.0}
        }
    ]
    
    results = []
    for req in requests:
        context = ExecutionContext(
            params={
                "agent_id": req["agent_id"],
                "resource_request": req["request"]
            },
            state={},
            metadata={"timestamp": 0.0}
        )
        result = await resource_instruction.execute(context)
        results.append(result)
    
    # Verify results
    for result in results:
        assert result.success
        assert "allocation" in result.output
    
    # Verify resource tracking
    usage = resource_instruction._calculate_usage()
    assert usage["cpu"] == 70.0
    assert usage["memory"] == 768.0
    assert usage["disk"] == 3000.0
    
    # Verify metrics
    assert resource_instruction.metrics.coordination_attempts == len(requests)
    assert resource_instruction.metrics.successful_coordinations == len(requests)
    assert 0.0 <= resource_instruction.metrics.resource_utilization <= 1.0

@pytest.mark.asyncio
async def test_resource_overallocation(resource_instruction):
    """Test resource overallocation prevention."""
    # First allocation
    context1 = ExecutionContext(
        params={
            "agent_id": "agent1",
            "resource_request": {"cpu": 90.0, "memory": 900.0, "disk": 9000.0}
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    result1 = await resource_instruction.execute(context1)
    assert result1.success
    
    # Second allocation (should fail)
    context2 = ExecutionContext(
        params={
            "agent_id": "agent2",
            "resource_request": {"cpu": 20.0, "memory": 200.0, "disk": 2000.0}
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    result2 = await resource_instruction.execute(context2)
    
    assert not result2.success
    assert any("insufficient" in str(error["error"]).lower() for error in result2.errors)
    assert resource_instruction.metrics.failed_coordinations > 0

@pytest.mark.asyncio
async def test_workflow_coordination(workflow_instruction):
    """Test workflow coordination."""
    workflow_id = "test_workflow_1"
    agent_id = "agent1"
    
    # Start workflow
    start_context = ExecutionContext(
        params={
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "action": "start"
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    start_result = await workflow_instruction.execute(start_context)
    assert start_result.success
    assert start_result.output["state"] == "running"
    
    # Update workflow
    update_context = ExecutionContext(
        params={
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "action": "update",
            "action_data": {
                "progress": 0.5,
                "status": "processing"
            }
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    update_result = await workflow_instruction.execute(update_context)
    assert update_result.success
    assert "progress" in update_result.output["data"]
    
    # Complete workflow
    complete_context = ExecutionContext(
        params={
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "action": "complete"
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    complete_result = await workflow_instruction.execute(complete_context)
    assert complete_result.success
    assert complete_result.output["state"] == "completed"
    
    # Verify workflow history
    workflow = workflow_instruction.active_workflows[workflow_id]
    assert len(workflow["actions"]) == 3
    assert workflow["state"] == "completed"
    
    # Verify metrics
    assert workflow_instruction.metrics.coordination_attempts == 3
    assert workflow_instruction.metrics.successful_coordinations == 3
    assert workflow_instruction.metrics.failed_coordinations == 0

@pytest.mark.asyncio
async def test_invalid_workflow_actions(workflow_instruction):
    """Test invalid workflow action handling."""
    workflow_id = "test_workflow_2"
    agent_id = "agent1"
    
    # Try to complete before starting
    context = ExecutionContext(
        params={
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "action": "complete"
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await workflow_instruction.execute(context)
    
    assert not result.success
    assert any("invalid" in str(error["error"]).lower() for error in result.errors)
    assert workflow_instruction.metrics.failed_coordinations > 0

@pytest.mark.asyncio
async def test_multi_agent_workflow(workflow_instruction):
    """Test multi-agent workflow coordination."""
    workflow_id = "test_workflow_3"
    agents = ["agent1", "agent2", "agent3"]
    
    # Start workflow
    start_context = ExecutionContext(
        params={
            "workflow_id": workflow_id,
            "agent_id": agents[0],
            "action": "start"
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    await workflow_instruction.execute(start_context)
    
    # Multiple agents update workflow
    for agent_id in agents:
        update_context = ExecutionContext(
            params={
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "action": "update",
                "action_data": {
                    "progress": 0.3,
                    "status": f"{agent_id}_processing"
                }
            },
            state={},
            metadata={"timestamp": 0.0}
        )
        result = await workflow_instruction.execute(update_context)
        assert result.success
    
    # Verify workflow state
    workflow = workflow_instruction.active_workflows[workflow_id]
    assert len(workflow["participants"]) == len(agents)
    assert len(workflow["actions"]) == len(agents) + 1  # start + updates
    
    # Verify data updates
    for agent_id in agents:
        assert f"{agent_id}_processing" in str(workflow["data"]) 

# ----------------------------------------------------------------------
# Supplementary edge-case tests
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_barrier_width_from_expected_participants(coordination_config):
    """Barrier width can be pinned per call via expected_participants."""
    instruction = AgentSynchronizationInstruction(
        name="width_sync", coordination_config=coordination_config
    )

    async def arrive(agent_id, expected):
        ctx = ExecutionContext(
            params={
                "sync_id": "width-1",
                "agent_id": agent_id,
                "sync_data": {"who": agent_id},
                "expected_participants": expected,
            },
            state={},
            metadata={},
        )
        return await instruction.execute(ctx)

    # Two agents, barrier width explicitly 2: both release together.
    r1, r2 = await asyncio.gather(arrive("a1", 2), arrive("a2", 2))
    assert r1.success and r2.success
    shared = r1.output["sync_data"]
    assert {v["who"] for v in shared.values()} == {"a1", "a2"}


@pytest.mark.asyncio
async def test_single_agent_barriers_on_width_one(coordination_config):
    """A single agent with expected_participants=1 releases immediately."""
    instruction = AgentSynchronizationInstruction(
        name="solo_sync", coordination_config=coordination_config
    )
    ctx = ExecutionContext(
        params={
            "sync_id": "solo-1",
            "agent_id": "lone",
            "sync_data": {},
            "expected_participants": 1,
        },
        state={},
        metadata={},
    )
    result = await instruction.execute(ctx)
    assert result.success


@pytest.mark.asyncio
async def test_barrier_timeout_when_peer_missing(coordination_config):
    """An agent waiting on a width-2 barrier with no peer must time out."""
    fast_config = CoordinationConfig(timeout_seconds=0.2)
    instruction = AgentSynchronizationInstruction(
        name="timeout_sync", coordination_config=fast_config
    )
    ctx = ExecutionContext(
        params={
            "sync_id": "lonely-1",
            "agent_id": "waiting",
            "sync_data": {},
            "expected_participants": 2,
        },
        state={},
        metadata={},
    )
    result = await instruction.execute(ctx)
    assert not result.success
    assert any("timeout" in str(e["error"]).lower() for e in result.errors)
    assert instruction.metrics.failed_coordinations == 1


@pytest.mark.asyncio
async def test_concurrent_resource_requests(coordination_config):
    """Concurrent requests are limit-checked; only fitting ones succeed."""
    instruction = ResourceCoordinationInstruction(
        name="conc_resources",
        coordination_config=coordination_config,
        resource_limits={"cpu": 8.0},
    )

    async def request(agent_id, amount):
        ctx = ExecutionContext(
            params={"agent_id": agent_id, "resource_request": {"cpu": amount}},
            state={},
            metadata={},
        )
        return await instruction.execute(ctx)

    results = await asyncio.gather(*[request(f"a{i}", 2.0) for i in range(5)])
    # Exactly four 2.0-CPU requests fit within the limit of 8.0.
    assert sum(1 for r in results if r.success) == 4
    assert instruction.metrics.failed_coordinations == 1


@pytest.mark.asyncio
async def test_double_start_rejected(coordination_config):
    """A second start on a running workflow is an invalid transition."""
    instruction = WorkflowCoordinationInstruction(
        name="ds_workflow", coordination_config=coordination_config, workflow_schema={}
    )
    ctx1 = ExecutionContext(
        params={"workflow_id": "wf-ds", "agent_id": "a1", "action": "start"},
        state={},
        metadata={},
    )
    ctx2 = ExecutionContext(
        params={"workflow_id": "wf-ds", "agent_id": "a2", "action": "start"},
        state={},
        metadata={},
    )
    assert (await instruction.execute(ctx1)).success
    assert not (await instruction.execute(ctx2)).success


@pytest.mark.asyncio
async def test_research_team_scenario(coordination_config):
    """Cross-instruction multi-agent scenario.

    Three agents (1) meet at a barrier sharing findings, (2) claim
    compute from a shared resource pool, and (3) drive a joint
    workflow to completion.
    """
    sync = AgentSynchronizationInstruction("sync", coordination_config)
    resources = ResourceCoordinationInstruction(
        "resources", coordination_config, resource_limits={"gpu": 4.0}
    )
    workflow = WorkflowCoordinationInstruction(
        "workflow", coordination_config, workflow_schema={}
    )

    # Stage 1: barrier with all three agents.
    def sync_ctx(i):
        return ExecutionContext(
            params={
                "sync_id": "research-sync",
                "agent_id": f"researcher-{i}",
                "sync_data": {"findings": i},
                "expected_participants": 3,
            },
            state={},
            metadata={},
        )

    findings = await asyncio.gather(*[sync.execute(sync_ctx(i)) for i in range(3)])
    assert all(r.success for r in findings)
    shared = findings[0].output["sync_data"]
    assert {d["findings"] for d in shared.values()} == {0, 1, 2}

    # Stage 2: shared resource pool — 3 x 1.0 GPU within limit 4.0.
    def res_ctx(i):
        return ExecutionContext(
            params={
                "agent_id": f"researcher-{i}",
                "resource_request": {"gpu": 1.0},
            },
            state={},
            metadata={},
        )

    allocations = await asyncio.gather(*[resources.execute(res_ctx(i)) for i in range(3)])
    assert all(r.success for r in allocations)
    assert resources.metrics.resource_utilization == pytest.approx(0.75)

    # Stage 3: joint workflow lifecycle.
    def wf_ctx(i, action, data=None):
        params = {
            "workflow_id": "research",
            "agent_id": f"researcher-{i}",
            "action": action,
        }
        if data is not None:
            params["action_data"] = data
        return ExecutionContext(params=params, state={}, metadata={})

    assert (await workflow.execute(wf_ctx(0, "start"))).success
    for i in range(3):
        result = await workflow.execute(wf_ctx(i, "update", {"contributions": i + 1}))
        assert result.success
    final = await workflow.execute(wf_ctx(0, "complete"))
    assert final.success
    assert final.output["state"] == "completed"
    participants = workflow.active_workflows["research"]["participants"]
    assert participants == {"researcher-0", "researcher-1", "researcher-2"}
