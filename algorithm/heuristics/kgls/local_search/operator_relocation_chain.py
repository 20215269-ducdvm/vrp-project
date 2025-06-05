from collections import defaultdict
import logging

from datastructure import Node, Route, Edge, VRPSolution, CostEvaluator
from .local_search_move import LocalSearchMove

# TODO continue valid chains to find even better improvements

logger = logging.getLogger(__name__)


class Relocation:

    def __init__(
            self,
            node_to_move: Node,
            cur_prev: Node,
            cur_next: Node,
            move_from_route: Route,
            move_to_route: Route,
            move_after: Node,
            move_before: Node,
            improvement: float
    ):
        self.node_to_move = node_to_move
        self.move_from_route = move_from_route
        self.move_to_route = move_to_route
        self.cur_prev = cur_prev
        self.cur_next = cur_next
        self.move_after = move_after
        self.move_before = move_before
        self.improvement = improvement
        self.forbidden_nodes = {
            node_to_move,
            cur_prev,
            cur_next,
            move_after,
            move_before
        }

    def __lt__(self, other):
        return self.improvement > other.improvement


class RelocationChain(LocalSearchMove):

    def __init__(self):
        self.relocations: list[Relocation] = []
        self.forbidden_nodes: set[Node] = set()
        # TODO better solution for below two things?
        # TODO tuple instead of edge?
        self.forbidden_insertion: set[Edge] = set()
        self.relocated_nodes: set[Node] = set()
        self.improvement: float = 0
        self.demand_changes: defaultdict[Route, int] = defaultdict(int)

    def get_routes(self) -> set[Route]:
        involved_routes = set()
        for relocation in self.relocations:
            involved_routes.add(relocation.move_to_route)
            involved_routes.add(relocation.move_from_route)
        return involved_routes

    def _add_relocation(self, relocation: Relocation):
        self.relocations.append(relocation)

        self.forbidden_nodes = self.forbidden_nodes | relocation.forbidden_nodes
        self.forbidden_insertion.add(Edge(relocation.move_after, relocation.move_before))
        # TODO not allowed to place after or before the relocated node in the old route
        self.forbidden_insertion.add(Edge(relocation.cur_prev, relocation.node_to_move))
        self.forbidden_insertion.add(Edge(relocation.node_to_move, relocation.cur_next))

        self.demand_changes[relocation.move_from_route] -= relocation.node_to_move.demand
        self.demand_changes[relocation.move_to_route] += relocation.node_to_move.demand

        self.relocated_nodes.add(relocation.node_to_move)
        self.improvement += relocation.improvement

    def can_insert_between(self, node1: Node, node2: Node):
        return (
                Edge(node1, node2) not in self.forbidden_insertion and
                node1 not in self.relocated_nodes and
                node2 not in self.relocated_nodes
        )

    def is_disjunct(self, other):
        for route in self.get_routes():
            if route in other.get_routes():
                return False
        return True

    def extend(self, relocation: Relocation):
        extended_chain = RelocationChain()
        extended_chain.relocations = self.relocations.copy()
        extended_chain.relocated_nodes = self.relocated_nodes.copy()
        extended_chain.forbidden_nodes = self.forbidden_nodes.copy()
        extended_chain.forbidden_insertion = self.forbidden_insertion.copy()
        extended_chain.improvement = self.improvement
        extended_chain.demand_changes = self.demand_changes.copy()

        extended_chain._add_relocation(relocation)

        return extended_chain

    def execute(self, solution: VRPSolution):
        logging.debug(
            f'Executing relocation with {len(self.relocations)} relocations '
            f'and improvement of {int(self.improvement)}')

        for relocation in self.relocations:
            solution.remove_nodes([relocation.node_to_move])
            solution.insert_nodes_after([relocation.node_to_move], relocation.move_after, relocation.move_to_route)

    def print_log(self):
        print(
            f'RelocationChain: {len(self.relocations)} relocations '
            f'with improvement of {int(self.improvement)}'
        )
        for relocation in self.relocations:
            print(
                f'  Relocation: {relocation.node_to_move.node_id} '
                f'from route {relocation.move_from_route.route_index} '
                f'to route {relocation.move_to_route.route_index}, '
                f'after node {relocation.move_after.node_id} '
                f'before node {relocation.move_before.node_id} '
                f'improvement: {int(relocation.improvement)}'
            )

def insert_node(
        node_to_move: Node,
        removal_gain: float,
        insert_next_to: Node,
        cur_chain: RelocationChain,
        solution: VRPSolution,
        cost_evaluator: CostEvaluator,
):
    # TODO check insertion before and after
    #insertion_cost = cost_evaluator.insertion_costs[node_to_move, insert_next_to]
    #insert_after = cost_evaluator.insertion_after[node_to_move, insert_next_to]
    predecessor = solution.prev(insert_next_to)
    successor = solution.next(insert_next_to)
    insertion_cost_before = (
            cost_evaluator.get_distance(node_to_move, predecessor)
            + cost_evaluator.get_distance(node_to_move, insert_next_to)
            - cost_evaluator.get_distance(predecessor, insert_next_to)
    )
    insertion_cost_after = (
            cost_evaluator.get_distance(node_to_move, successor)
            + cost_evaluator.get_distance(node_to_move, insert_next_to)
            - cost_evaluator.get_distance(successor, insert_next_to)
    )
    if insertion_cost_before <= insertion_cost_after:
        insertion_cost = insertion_cost_before
        insert_after = predecessor
        insert_before = insert_next_to
    else:
        insertion_cost = insertion_cost_after
        insert_after = insert_next_to
        insert_before = successor

    cost_change = removal_gain - insertion_cost

    if cur_chain.improvement + cost_change > 0:
        if cur_chain.can_insert_between(insert_after, insert_before):
            route = solution.route_of(insert_next_to)

            return Relocation(
                node_to_move=node_to_move,
                cur_prev=solution.prev(node_to_move),
                cur_next=solution.next(node_to_move),
                move_from_route=solution.route_of(node_to_move),
                move_to_route=route,
                move_after=insert_after,
                move_before=insert_before,
                improvement=cost_change
            )

    return None


def search_relocation_chains_from(
        valid_relocations_chain: list,
        solution: VRPSolution,
        cost_evaluator: CostEvaluator,
        node_to_move: Node,
        max_depth: int,
        current_depth: int = 0,
        cur_chain: RelocationChain = None
):
    if current_depth >= max_depth:
        # Stopping condition: Maximum depth reached
        return None

    # initialize the first chain
    if cur_chain is None:
        cur_chain = RelocationChain()

    # Step 1: Calculate the cost change from removing the node
    cur_prev = solution.prev(node_to_move)
    cur_next = solution.next(node_to_move)
    removal_improvement = (
            cost_evaluator.get_distance(node_to_move, cur_prev)
            + cost_evaluator.get_distance(node_to_move, cur_next)
            - cost_evaluator.get_distance(cur_prev, cur_next)
    )

    # Step 2: For each candidate neighbour of 'node_to_move',
    # check whether a relocation next to it would improve the solution
    from_route = solution.route_of(node_to_move)
    # Get the cost of the source route

    candidate_insertions = defaultdict(list)
    for neighbour in cost_evaluator.get_neighborhood(node_to_move):
        to_route = solution.route_of(neighbour)

        if to_route != from_route and neighbour not in cur_chain.relocated_nodes:
            insertion = insert_node(
                node_to_move=node_to_move,
                removal_gain=removal_improvement,
                insert_next_to=neighbour,
                cur_chain=cur_chain,
                solution=solution,
                cost_evaluator=cost_evaluator
            )
            if insertion:
                candidate_insertions[to_route].append(
                    insertion
                )

    for destination_route, insertions in candidate_insertions.items():
        best_insertion = sorted(insertions)[0]
        extended_chain = cur_chain.extend(best_insertion)

        # Get the cost of the destination route
        dist2, load2, tw2 = solution.get_route_penalties(destination_route)
        to_route_cost = cost_evaluator.get_route_cost(dist2, load2, tw2)

        # Create proposal for source route with the node removed
        proposal_from_route = from_route.proposal(
            before_segment=from_route.before(from_route.idx_of[cur_prev.node_id]),
            middle_segments=[],  # Skip the node_to_move
            after_segment=from_route.after(from_route.idx_of[cur_next.node_id])
        )

        # Create proposal for destination route with the node inserted

        # Create the proposal with the node inserted
        proposal_to_route = destination_route.proposal(
            before_segment=destination_route.before(destination_route.idx_of[best_insertion.move_after.node_id]),
            middle_segments=[
                # Add the node_to_move as a segment
                # We'll simulate this by using a dummy SegmentMiddle with just the node
                from_route.at(from_route.idx_of[node_to_move.node_id])
            ],
            after_segment=destination_route.after(destination_route.idx_of[best_insertion.move_before.node_id])
        )

        # Get cost of the destination route proposal
        proposal_to_cost = cost_evaluator.get_route_cost(
            proposal_to_route.penalties[0],
            proposal_to_route.penalties[1],
            proposal_to_route.penalties[2]
        )

        # Check if the target route after proposal is feasible
        if (
                proposal_from_route.is_feasible and
                proposal_to_route.is_feasible
        ) or proposal_to_cost > to_route_cost:
            valid_relocations_chain.append(extended_chain)
        else:
            if len(extended_chain.relocations) < max_depth:
                # try to restore feasibility by a follow-up relocation
                for candidate_node in destination_route.customers:
                    if candidate_node not in extended_chain.forbidden_nodes:
                        # Create a proposal with candidate_node removed from destination route
                        candidate_prev = solution.prev(candidate_node)
                        candidate_next = solution.next(candidate_node)

                        proposal_candidate_removed = destination_route.proposal(
                            before_segment=destination_route.before(destination_route.idx_of[candidate_prev.node_id]),
                            middle_segments=[
                                # Add the node_to_move to replace the removed candidate
                                from_route.at(from_route.idx_of[node_to_move.node_id])
                            ],
                            after_segment=destination_route.after(destination_route.idx_of[candidate_next.node_id])
                        )

                        proposal_with_candidate_removed_cost = cost_evaluator.get_route_cost(
                            proposal_candidate_removed.penalties[0],
                            proposal_candidate_removed.penalties[1],
                            proposal_candidate_removed.penalties[2]
                        )

                        is_improved = (
                            proposal_candidate_removed.is_feasible or
                            proposal_with_candidate_removed_cost > to_route_cost
                        )

                        if is_improved:
                            search_relocation_chains_from(
                                valid_relocations_chain=valid_relocations_chain,
                                solution=solution,
                                cost_evaluator=cost_evaluator,
                                node_to_move=candidate_node,
                                max_depth=max_depth,
                                current_depth=current_depth + 1,
                                cur_chain=extended_chain
                            )


def search_relocation_chains(
        solution: VRPSolution,
        cost_evaluator: CostEvaluator,
        start_nodes: list[Node],
        max_depth: int,
) -> list[RelocationChain]:
    found_moves = []
    for start_node in start_nodes:
        search_relocation_chains_from(
            valid_relocations_chain=found_moves,
            solution=solution,
            cost_evaluator=cost_evaluator,
            node_to_move=start_node,
            max_depth=max_depth
        )
    return sorted(found_moves)
