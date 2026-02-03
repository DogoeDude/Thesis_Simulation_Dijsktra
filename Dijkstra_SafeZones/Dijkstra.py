
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

def calculate_dijkstra_repair(G, center_id, radius_meters, current_dist_map, sources=None):
    """
    Performs an Anchor-Based Local Repair of the Dijkstra Field.
    
    Logic:
    1. Find 'Affected Zone' (S) = Nodes within radius of center.
    2. Find 'Anchors' (B) = Nodes NOT in S, but connected to S.
    3. Seed Dijkstra with Anchors using their EXISTING/VALID current_dist_map values.
    4. ALSO Seed with any SOURCES that act as 0-distance sinks (Exits/SafeZones) inside S.
    5. Propagate inwards to repair S.
    """
    # 1. Identify Scope (BFS to find nodes within radius)
    # Note: If graph weights are distances, we can use simple BFS if edges are roughly uniform,
    # or a quick Dijkstra check. For speed, we'll use geometric distance if available, or hop count.
    # Let's use Graph Traversal for "Network Radius".
    
    affected_zone = set()
    q = [(0, center_id)]
    visited_scope = {center_id: 0}
    
    # Quick collection of zone
    zone_nodes = []
    
    # We will simply iterate ALL nodes for geometric check if graph is small (<1000 nodes), 
    # OR use BFS if large. Since N~300, iteration is instant.
    center_pt = G.nodes[center_id]['geometry'].centroid
    
    for n in G.nodes:
        # Check dist
        pt = G.nodes[n]['geometry'].centroid
        if pt.distance(center_pt) <= radius_meters:
            affected_zone.add(n)
            
    # 2. Identify Anchors (Rim)
    anchors = []
    for n in affected_zone:
        for neighbor in G.neighbors(n):
            if neighbor not in affected_zone:
                # This neighbor is an Anchor
                anchor_cost = current_dist_map.get(neighbor, float('inf'))
                if anchor_cost != float('inf'):
                    anchors.append(neighbor)
    
    anchors = list(set(anchors)) # Unique
    
    # 3. Setup Dijkstra
    # We only want to solve for nodes IN affected_zone.
    # But we seed with Anchors.
    
    distances = {n: float('inf') for n in affected_zone}
    
    # Seed inputs
    pq = []
    
    # Add Anchors to PQ
    for anc in anchors:
        cost = current_dist_map[anc] # Trust the outside world
        # We don't add Anchor to 'distances' dict because we don't want to output it (it's unchanged)
        # But we push it to PQ so it propagates to neighbors
        heapq.heappush(pq, (cost, anc))
        
    # CRITICAL FIX: Re-seed Sources inside the zone !
    if sources:
        for s in sources:
            if s in affected_zone:
                # Source cost is typically 0 (or low). 
                # Let's assume it's 0 OR take from current map if we want initial costs preserved.
                cost = 0.0
                distances[s] = cost
                heapq.heappush(pq, (cost, s))
    
    # 4. Run Logic
    final_updates = {}
    
    while pq:
        d, u = heapq.heappop(pq)
        
        # If u is in affected_zone, we record it
        if u in affected_zone:
            if d < distances[u]:
                distances[u] = d
                final_updates[u] = d
            else:
                continue # Already found better path to this zone node
        
        # Propagate
        for v in G.neighbors(u):
            # We only care about propagating INTO or WITHIN the affected zone
            if v in affected_zone:
                weight = G[u][v].get('weight', 1.0)
                new_dist = d + weight
                
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    final_updates[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
                    
    return final_updates


def calculate_rdd_repair(G, hazard_node, current_dist_map):
    """
    Implements Inconsistency-Based (RDD) Local Repair.
    
    Logic:
    1. Identify 'Locally Inconsistent' nodes (Downstream Branch).
       - Start at hazard_node.
       - Find neighbors who relied on hazard_node (Dist[n] ≈ Dist[h] + w).
       - Recursively findings nodes who relied on them.
    2. Reset this 'Disturbed Branch' to Infinity.
    3. Identify Anchors:
       - Neighbors of the Disturbed Branch that are NOT disturbed.
    4. Propagate Repair Wave from Anchors.
    
    Args:
        G: NetworkX graph
        hazard_node: The node ID where the penalty increased.
        current_dist_map: The existing (now partially invalid) distance map.
        
    Returns:
        dict: {node_id: new_distance} (Only contains updated nodes)
    """
    # 1. IDENTIFY DISTURBED BRANCH (Downstream Dependency Tracing)
    # We need to find nodes whose specific path went through the hazard.
    # Since we don't store predecessors, we infer them:
    # If Dist[n] == Dist[parent] + Weight(parent, n), then n depends on parent.
    
    disturbed_set = set() 
    
    # Queue for BFS dependency tracing
    # We assume 'hazard_node' ITSELF is the start of the disturbance (it just got expensive).
    # Actually, the hazard_node itself might be fine if it can go elsewhere, 
    # but initially, nodes *upstream* flowing *into* it are the problem.
    # Wait, Reverse Dijkstra: Path flows Sink -> Source.
    # Agents flow Source -> Sink.
    # Distance represents "Cost to Sink".
    # If Hazard Node cost increases, then nodes that go TO Hazard Node (neighbors) have increased cost.
    # So "Downstream" in the algorithms sense (Dependency) is "Upstream" in traffic flow.
    disturbed_set.add(hazard_node)
    open_list = [hazard_node]
    
    # We need to check checks relative to the "Old" state, but we only have current weights.
    # The Inconsistency: Dist[n] < Dist[best_neighbor] + new_weight
    # Actually, simplistic tracing: Look for any neighbor n where:
    # Dist[n] was derived from current `item`?
    # HEURISTIC: Check if Dist[n] > Dist[item]. If so, it *could* correspond to a path item -> n?
    # NO. In reverse dijkstra (Cost to Exit), flow is n -> item -> ... -> Exit.
    # So Dist[n] = Dist[item] + weight.
    # So Dist[n] > Dist[item].
    # So we search for neighbors 'n' where Dist[n] ≈ Dist[item] + Weight(n, item).
    
    while open_list:
        curr = open_list.pop(0)
        curr_dist = current_dist_map.get(curr, float('inf'))
        
        for n in G.neighbors(curr):
            if n in disturbed_set: continue
            
            # Check if n depends on curr
            # Note: Weights might have just changed. We need to be careful.
            # If we are checking the hazard node itself, the edge weight (n, curr) might have increased.
            # If Dist[n] was based on the old weight, checking with NEW weight might mismatch.
            # BUT, generally: Dist[n] should be GREATER than Dist[curr].
            
            n_dist = current_dist_map.get(n, float('inf'))
            
            # Optimization: If n is an exit/sink (Dist=0), it definitely doesn't depend on curr.
            if n_dist == 0: continue
            
            # Weighted Edge
            w = G[curr][n].get('weight_original', G[curr][n].get('weight', 1.0))
            # (Use original weight for dependency check? No, the distances were built on whatever was there).
            # Let's use the CURRENT effective weight in the map to be safe, or just geometry.
            # Actually, strictest check:
            # If n_dist > curr_dist, 'n' might be flowing into 'curr'.
            
            if n_dist > curr_dist:
                # Potential dependency. 
                # Strict check: is it "close" to the edge weight?
                # Allow a small epsilon for float errors
                if n_dist <= (curr_dist + w * 2.0): # Relaxed check: Is it reasonably "local"?
                    # To include it in the "Disturbed Branch", we are essentially saying:
                    # "We suspect you rely on 'curr'. We will force you to re-check."
                    disturbed_set.add(n)
                    open_list.append(n)
                    
    # 2. IDENTIFY ANCHORS & RESET
    # Anchors are neighbors of Disturbed Set that are NOT Disturbed.
    anchors = []
    
    for u in disturbed_set:
        for v in G.neighbors(u):
            if v not in disturbed_set:
                # v is an Anchor (Stable Node)
                # Ensure it has a valid distance
                d_v = current_dist_map.get(v, float('inf'))
                if d_v != float('inf'):
                    anchors.append(v)
                    
    anchors = list(set(anchors))
    
    # 3. REPAIR WAVE (Localized Dijkstra)
    # Initialize PQ with Anchors + Boundary Edges
    pq = []
    
    # We want to solve for nodes in disturbed_set.
    # The "Sources" for this sub-problem are the Anchors.
    
    # Initial Updates Dict
    repaired_values = {}
    
    # Seed PQ with Anchors (propagating TO disturbed nodes)
    for anc in anchors:
        anc_dist = current_dist_map[anc]
        
        # Try to push into neighbors that are in disturbed_set
        for n in G.neighbors(anc):
            if n in disturbed_set:
                weight = G[anc][n].get('weight', 1.0)
                new_dist = anc_dist + weight
                heapq.heappush(pq, (new_dist, n))

    # Also, we must handle the case where the Hazard Node ITSELF finds a new path?
    # The Hazard Node is in disturbed_set. Anchors will feed it.
    
    # Run Dijkstra confined to Disturbed Set
    # We treat 'repaired_values' as the local distance map for disturbed nodes.
    # Initial state of disturbed nodes is effectively Infinity (unknown).
    
    while pq:
        d, u = heapq.heappop(pq)
        
        # We only process if u is in disturbed_set
        if u not in disturbed_set: continue
        
        # Optimization: Check if we found a better path already
        if u in repaired_values and repaired_values[u] <= d:
            continue
            
        # Commit
        repaired_values[u] = d
        
        # Propagate to neighbors within disturbed_set
        for v in G.neighbors(u):
            if v in disturbed_set:
                weight = G[u][v].get('weight', 1.0)
                new_d = d + weight
                
                # Check if this beats current known best
                if v not in repaired_values or new_d < repaired_values[v]:
                    heapq.heappush(pq, (new_d, v))
                    
    return repaired_values
