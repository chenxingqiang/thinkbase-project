"""Ell integration for AgentFlow monitoring and visualization."""
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import sqlite3
import json

from agentflow.ell.integration import ELLIntegration
from agentflow.core.base_types import WorkflowStatus

class EllMonitor:
    """Ell monitoring integration for AgentFlow."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ell monitor.
        
        Args:
            config: Configuration dictionary containing:
                - storage_dir: Directory for storing Ell data
                - project_name: Name of the project
                - version: Version of the project
        """
        self.config = config
        self.storage_dir = config.get('storage_dir', './ell_data')
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize SQLite database
        self.db_path = os.path.join(self.storage_dir, 'traces.db')
        self._init_db()
        
        # Initialize Ell
        self.ell = ELLIntegration()
        self.ell.configure({
            'project': config.get('project_name', 'agentflow'),
            'version': config.get('version', '1.0.0'),
            'model': 'deepseek-reasoner-v1'
        })
        
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS agent_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                prompt TEXT,
                completion TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS workflow_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT,
                steps TEXT,
                metadata TEXT,
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def track_agent_execution(self, agent_id: str, prompt: str, completion: str,
                            metadata: Optional[Dict[str, Any]] = None):
        """Track agent execution in database.
        
        Args:
            agent_id: ID of the agent
            prompt: Input prompt
            completion: Model completion
            metadata: Optional metadata
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute(
            'INSERT INTO agent_executions (agent_id, prompt, completion, metadata) VALUES (?, ?, ?, ?)',
            (agent_id, prompt, completion, json.dumps(metadata or {}))
        )
        
        conn.commit()
        conn.close()
        
    def track_workflow_execution(self, workflow_id: str, steps: List[Dict[str, Any]],
                               metadata: Optional[Dict[str, Any]] = None,
                               status: WorkflowStatus = WorkflowStatus.PENDING):
        """Track workflow execution in database.
        
        Args:
            workflow_id: ID of the workflow
            steps: List of workflow steps
            metadata: Optional metadata
            status: Current workflow status
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute(
            'INSERT INTO workflow_executions (workflow_id, steps, metadata, status) VALUES (?, ?, ?, ?)',
            (workflow_id, json.dumps(steps), json.dumps(metadata or {}), status.value)
        )
        
        conn.commit()
        conn.close()
        
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary containing agent metrics
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM agent_executions WHERE agent_id = ?', (agent_id,))
        total_executions = c.fetchone()[0]
        
        c.execute('SELECT AVG(LENGTH(completion)) FROM agent_executions WHERE agent_id = ?', (agent_id,))
        avg_completion_length = c.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_executions': total_executions,
            'avg_completion_length': avg_completion_length
        }
        
    def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get metrics for a specific workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Dictionary containing workflow metrics
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM workflow_executions WHERE workflow_id = ?', (workflow_id,))
        total_executions = c.fetchone()[0]
        
        c.execute('SELECT status, COUNT(*) FROM workflow_executions WHERE workflow_id = ? GROUP BY status',
                 (workflow_id,))
        status_counts = dict(c.fetchall())
        
        conn.close()
        
        return {
            'total_executions': total_executions,
            'status_counts': status_counts
        }
        
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization.
        
        Returns:
            Dictionary containing visualization data
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get agent execution timeline
        c.execute('''
            SELECT agent_id, COUNT(*), strftime('%Y-%m-%d', timestamp) as date
            FROM agent_executions 
            GROUP BY agent_id, date
            ORDER BY date
        ''')
        agent_timeline = c.fetchall()
        
        # Get workflow execution timeline
        c.execute('''
            SELECT workflow_id, status, COUNT(*), strftime('%Y-%m-%d', timestamp) as date
            FROM workflow_executions
            GROUP BY workflow_id, status, date
            ORDER BY date
        ''')
        workflow_timeline = c.fetchall()
        
        conn.close()
        
        return {
            'agent_timeline': agent_timeline,
            'workflow_timeline': workflow_timeline
        }

if __name__ == '__main__':
    # Example usage
    config = {
        'storage_dir': './ell_data',
        'project_name': 'agentflow',
        'version': '1.0.0'
    }
    
    monitor = EllMonitor(config)
    print("ELL Monitor initialized successfully!")
    print(f"Data directory: {monitor.storage_dir}")
    print(f"Database path: {monitor.db_path}")
