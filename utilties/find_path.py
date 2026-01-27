def find_path(neighbors_function, start_state, goal_test, bfs=True):
    """
        @param neighbors_function: neighbors_function(state) is a function which returns a list of legal
            neighbor states
        @param start_state: is the starting state for the search
        @param goal_test: is a function which returns True if the given state is
            a goal state for the search, and False otherwise.
        @param bfs: is a boolean (default True) that indicates whether we should run a
    bfs or dfs
        @return: a path through a graph from a given starting state to any state that
        satisfies a given goal condition (or None if no such path exists).

            A path from start_state to a state satisfying goal_test(state) as a
            tuple of states, or None if no path exists.
        Note the state representation must be hashable in order for this function
        to work.
    """
    if goal_test(start_state):
        return (start_state,)
    agenda = [(start_state,)]
    visited = {start_state}
    while agenda:
        this_path = agenda.pop(0 if bfs else -1)
        terminal_state = this_path[-1]
        for neighbor_state in neighbors_function(terminal_state):
            if neighbor_state not in visited:
                new_path = this_path + (neighbor_state,)
                if goal_test(neighbor_state):
                    return new_path
                agenda.append(new_path)
                visited.add(neighbor_state)
