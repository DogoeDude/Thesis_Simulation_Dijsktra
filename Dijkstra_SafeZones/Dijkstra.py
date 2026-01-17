
import networkx as nx
import heapq

def calculate_dijkstra_field(G, exit_nodes, initial_costs=None, targets=None):
    """
    Calculates the distance from every node in the graph G to the NEAREST exit node
    using a Reverse Dijkstra (Multi-Source Dijkstra) approach.
    
    Args:
        G (nx.Graph): The networkx graph representing the road network.
        exit_nodes (list): A list of node IDs that are designated as exits/sources.
        initial_costs (dict): Optional {node: cost} to initialize sources with specific values.
        targets (set/list): Optional. If provided, the algorithm stops once ALL target nodes 
                            have been reached (visited).
        
    Returns:
        tuple: (distances_dict, visited_history_list)
    """
    
    # Initialize distances to infinity (or simply track visited)
    distances = {node: float('inf') for node in G.nodes()}
    
    # Priority Queue: (distance, node)
    pq = []
    
    # Initialize all exits with distance 0 (or initial_cost) and push to PQ
    for exit_node in exit_nodes:
        if exit_node in G:
            # Determine start cost
            cost = 0.0
            if initial_costs and exit_node in initial_costs:
                cost = initial_costs[exit_node]
            
            # Only update if better (unlikely to be worse than inf, but good practice)
            if cost < distances[exit_node]:
                distances[exit_node] = cost
                heapq.heappush(pq, (cost, exit_node))
            
    visited_order = [] # List of (node_id, distance) in order of settlement
    
    # Early Exit Logic
    target_set = set(targets) if targets else None
    targets_found = 0
    total_targets = len(target_set) if target_set else 0
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        # If we found a shorter path to u already, skip
        if current_dist > distances[u]:
            continue
        
        visited_order.append((u, current_dist))
        
        # Check Targets
        if target_set and u in target_set:
            # We just settled a target
            # Note: We track settled nodes, so we know this is the shortest path to this target
            targets_found += 1
            if targets_found >= total_targets:
                # OPTIMIZATION: We found all targets. 
                # Should we continue slightly to ensure neighbors are valid? 
                # Strictly for "connectivity", we can stop.
                break
        
        # Explore neighbors
        for v in G.neighbors(u):
            weight = G[u][v].get('weight', 1.0) 
            
            new_dist = current_dist + weight
            
            # Relaxation step
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
                
    return distances, visited_order
