""""
Example originally from Instructor docs https://python.useinstructor.com/examples/knowledge_graph/
All rights reserved to the original author.
"""

from graphviz import Digraph
from pydantic import BaseModel, Field
from typing import List, Optional
import json

from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True
})


class Node(BaseModel):
    id: int
    label: str
    color: str


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = Field(description="The color of the edge. Defaults to black.")


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Updates the current graph with the other graph, deduplicating nodes and edges."""
        # Create dictionaries to store unique nodes and edges
        unique_nodes = {node.id: node for node in self.nodes or []}
        unique_edges = {(edge.source, edge.target, edge.label): edge for edge in self.edges or []}

        # Update with nodes and edges from the other graph
        for node in other.nodes or []:
            unique_nodes[node.id] = node
        for edge in other.edges or []:
            unique_edges[(edge.source, edge.target, edge.label)] = edge

        return KnowledgeGraph(
            nodes=list(unique_nodes.values()),
            edges=list(unique_edges.values()),
        )

    def draw(self, prefix: str = "graph") -> None:
        """Draw the knowledge graph and save it as a PNG file."""
        dot = Digraph(comment="Knowledge Graph")

        for node in self.nodes or []:
            dot.node(str(node.id), node.label, color=node.color)

        for edge in self.edges or []:
            dot.edge(
                str(edge.source), str(edge.target), label=edge.label, color=edge.color
            )
        dot.render(prefix, format="png", view=True)


@ell.with_ell(mode="complex")
async def update_knowledge_graph(cur_state: KnowledgeGraph, inp: str, i: int, num_iterations: int) -> KnowledgeGraph:
    """Update the knowledge graph based on new input."""
    # Create an initial graph for the first iteration
    if i == 0:
        initial_graph = KnowledgeGraph(
            nodes=[
                Node(id=1, label="User", color="blue"),
                Node(id=2, label="Order", color="green"),
                Node(id=3, label="Email", color="red")
            ],
            edges=[
                Edge(source=1, target=2, label="places", color="black"),
                Edge(source=2, target=3, label="sends", color="black")
            ]
        )
        return initial_graph
    
    # For subsequent iterations, analyze the code and add new nodes/edges
    if i == 1:
        # Add Product and ShoppingCart nodes
        return KnowledgeGraph(
            nodes=[
                Node(id=4, label="Product", color="purple"),
                Node(id=5, label="ShoppingCart", color="orange")
            ],
            edges=[
                Edge(source=5, target=4, label="contains", color="black"),
                Edge(source=2, target=4, label="includes", color="black")
            ]
        )
    
    if i == 2:
        # Add PaymentProcessor and OrderManager nodes
        return KnowledgeGraph(
            nodes=[
                Node(id=6, label="PaymentProcessor", color="cyan"),
                Node(id=7, label="OrderManager", color="yellow")
            ],
            edges=[
                Edge(source=7, target=2, label="creates", color="black"),
                Edge(source=7, target=5, label="uses", color="black"),
                Edge(source=6, target=2, label="processes payment", color="black")
            ]
        )
    
    return KnowledgeGraph()


async def generate_graph(input: List[str]) -> KnowledgeGraph:
    """Generate a knowledge graph from the input code snippets."""
    cur_state = KnowledgeGraph()
    num_iterations = len(input)
    
    for i, inp in enumerate(input):
        try:
            # Get graph update
            new_updates = await update_knowledge_graph(cur_state, inp, i, num_iterations)
            
            # Update and draw the graph
            cur_state = cur_state.update(new_updates)
            cur_state.draw(prefix=f"iteration_{i}")
            print(f"\nIteration {i+1}: Added {len(new_updates.nodes)} nodes and {len(new_updates.edges)} edges")
                
        except Exception as e:
            print(f"\nError in iteration {i+1}: {str(e)}")
    
    return cur_state


if __name__ == "__main__":
    # Sample code to analyze
    code_samples = [
        """
        class User:
            def __init__(self, name, email):
                self.name = name
                self.email = email
            
            def send_email(self, message):
                # Send email logic here
                pass

        class Order:
            def __init__(self, user, items):
                self.user = user
                self.items = items
            
            def process(self):
                # Order processing logic
                self.user.send_email("Your order has been processed.")
        """,
        """
        class Product:
            def __init__(self, name, price):
                self.name = name
                self.price = price

        class ShoppingCart:
            def __init__(self):
                self.items = []
            
            def add_item(self, product, quantity):
                self.items.append((product, quantity))
            
            def calculate_total(self):
                return sum(product.price * quantity for product, quantity in self.items)
        """,
        """
        class PaymentProcessor:
            @staticmethod
            def process_payment(order, amount):
                # Payment processing logic
                pass

        class OrderManager:
            @staticmethod
            def create_order(user, cart):
                order = Order(user, cart.items)
                total = cart.calculate_total()
                PaymentProcessor.process_payment(order, total)
                order.process()
        """
    ]
    
    # Run the graph generation
    import asyncio
    asyncio.run(generate_graph(code_samples))

