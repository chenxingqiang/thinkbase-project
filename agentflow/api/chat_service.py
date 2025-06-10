"""Chat service for AgentFlow."""
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
import json
import asyncio
from ..agents.agent import Agent
from ..core.config import AgentConfig
from ..monitoring.ell_monitor import EllMonitor

class ChatService:
    """Chat service for handling real-time communication."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize chat service.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self.app = FastAPI()
        self.active_connections: List[WebSocket] = []
        self.agents: Dict[str, Agent] = {}
        self.ell_monitor = EllMonitor(config.get('ell_config', {}))
        
        # Register routes
        self._register_routes()
        
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.websocket("/chat")
        async def chat_endpoint(websocket: WebSocket):
            await self.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_json()
                    await self.handle_message(websocket, data)
            except WebSocketDisconnect:
                await self.disconnect(websocket)
                
        @self.app.post("/api/upload")
        async def upload_file(file: UploadFile = File(...)):
            # Handle file upload
            # Store file and return URL
            return {"url": f"/uploads/{file.filename}"}
            
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send available agents
        await websocket.send_json({
            "type": "init",
            "agents": [
                {
                    "id": agent_id,
                    "name": agent.config.agent["name"],
                    "type": agent.config.agent["type"],
                    "status": "ready"
                }
                for agent_id, agent in self.agents.items()
            ]
        })
        
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection.
        
        Args:
            websocket: WebSocket connection
        """
        self.active_connections.remove(websocket)
        
    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle incoming message.
        
        Args:
            websocket: WebSocket connection
            data: Message data
        """
        message_type = data.get("type")
        
        if message_type == "message":
            await self.process_chat_message(websocket, data)
        elif message_type == "command":
            await self.process_command(websocket, data)
            
    async def process_chat_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Process chat message.
        
        Args:
            websocket: WebSocket connection
            data: Message data
        """
        agent_id = data.get("agent_id")
        content = data.get("content")
        settings = data.get("settings", {})
        
        if not agent_id or not content:
            await websocket.send_json({
                "type": "error",
                "error": "Missing agent_id or content"
            })
            return
            
        agent = self.agents.get(agent_id)
        if not agent:
            await websocket.send_json({
                "type": "error",
                "error": f"Agent {agent_id} not found"
            })
            return
            
        try:
            # Track with Ell
            self.ell_monitor.track_agent_execution(
                agent_id=agent_id,
                prompt=content,
                completion="",  # Will be updated with response
                metadata={
                    "settings": settings,
                    "status": "processing"
                }
            )
            
            # Process message
            if settings.get("streamResponse"):
                async for chunk in agent.stream_process(content):
                    await websocket.send_json({
                        "type": "stream",
                        "content": chunk
                    })
                    
                    # Update Ell tracking
                    self.ell_monitor.track_agent_execution(
                        agent_id=agent_id,
                        prompt=content,
                        completion=chunk,
                        metadata={
                            "settings": settings,
                            "status": "streaming"
                        }
                    )
            else:
                response = await agent.process(content)
                await websocket.send_json({
                    "type": "response",
                    "content": response
                })
                
                # Update Ell tracking
                self.ell_monitor.track_agent_execution(
                    agent_id=agent_id,
                    prompt=content,
                    completion=response,
                    metadata={
                        "settings": settings,
                        "status": "completed"
                    }
                )
                
        except Exception as e:
            error_message = str(e)
            await websocket.send_json({
                "type": "error",
                "error": error_message
            })
            
            # Track error in Ell
            self.ell_monitor.track_agent_execution(
                agent_id=agent_id,
                prompt=content,
                completion="",
                metadata={
                    "settings": settings,
                    "status": "error",
                    "error": error_message
                }
            )
            
    async def process_command(self, websocket: WebSocket, data: Dict[str, Any]):
        """Process command message.
        
        Args:
            websocket: WebSocket connection
            data: Command data
        """
        command = data.get("command")
        
        if command == "clear_history":
            # Clear chat history
            pass
        elif command == "update_settings":
            # Update agent settings
            pass
            
    def register_agent(self, agent_id: str, agent: Agent):
        """Register new agent.
        
        Args:
            agent_id: Agent ID
            agent: Agent instance
        """
        self.agents[agent_id] = agent
        
    def start(self, host: str = "0.0.0.0", port: int = 8001):
        """Start chat service.
        
        Args:
            host: Service host
            port: Service port
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
        
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass
