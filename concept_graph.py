import math
import random
import json
from collections import defaultdict


class ConceptGraph:
    """A symbolic concept network with weighted, typed links.

    This is not a neural network: no layers, no backpropagation, no gradient updates.
    It is a graph-diffusion system where activation flows over concept relations.
    """

    def __init__(self, seed=7):
        random.seed(seed)
        self.nodes = set()
        self.edges = defaultdict(list)
        self.relation_gain = {
            "is_a": 0.95,
            "part_of": 0.9,
            "causes": 1.0,
            "evokes": 0.85,
            "associated_with": 0.75,
            "metaphor": 0.7,
            "contrast": -0.45,
            "cultural": 0.8,
            "sensory": 0.9,
            "inhibits": 1.0,
        }

    def add_node(self, name):
        self.nodes.add(name)

    def connect(self, a, b, weight, relation, bidirectional=True):
        self.add_node(a)
        self.add_node(b)
        self.edges[a].append((b, float(weight), relation))
        if bidirectional:
            self.edges[b].append((a, float(weight), relation))

    def _clamp(self, x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))

    def diffuse(
        self,
        seed_activation,
        steps=7,
        decay=0.2,
        tunnel_threshold=0.01,
        target_rms=0.42,
        soft_cap=0.92,
    ):
        """Spread activation through the idea graph.

        A light "tunneling-like" move is used when the network stalls:
        it jumps activation along strong 2-hop paths to escape local clusters.
        """
        activation = {n: 0.0 for n in self.nodes}
        for node, value in seed_activation.items():
            if node in activation:
                activation[node] = self._clamp(value)

        previous_total_delta = float("inf")

        for _ in range(steps):
            incoming = defaultdict(float)

            for src in self.nodes:
                src_act = activation[src]
                if abs(src_act) < 1e-6:
                    continue
                for dst, w, relation in self.edges[src]:
                    gain = self.relation_gain.get(relation, 0.7)
                    incoming[dst] += src_act * w * gain

            new_activation = {}
            total_delta = 0.0
            for node in self.nodes:
                raw = (1 - decay) * activation[node] + incoming[node]
                # Softsign keeps values responsive without hard tanh saturation.
                value = raw / (1 + abs(raw))
                new_activation[node] = self._clamp(value)
                total_delta += abs(new_activation[node] - activation[node])

            # Homeostatic normalization keeps graph-wide activity from collapsing to +1/-1.
            rms = math.sqrt(
                sum(v * v for v in new_activation.values()) / max(1, len(new_activation))
            )
            if rms > 1e-9:
                scale = min(1.0, target_rms / rms)
                for node in new_activation:
                    new_activation[node] = self._clamp(new_activation[node] * scale)

            # Final soft cap to preserve ranking while preventing total saturation.
            max_abs = max(abs(v) for v in new_activation.values()) if new_activation else 0.0
            if max_abs > soft_cap > 0:
                cap_scale = soft_cap / max_abs
                for node in new_activation:
                    new_activation[node] = self._clamp(new_activation[node] * cap_scale)

            # If movement stalls, push activation through best 2-hop routes.
            if abs(total_delta - previous_total_delta) < tunnel_threshold:
                self._tunnel_jump(new_activation, jump_scale=0.12)

            previous_total_delta = total_delta
            activation = new_activation

        return activation

    def _tunnel_jump(self, activation, jump_scale=0.12):
        candidates = sorted(activation.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for src, src_act in candidates:
            if abs(src_act) < 0.1:
                continue
            hops = []
            for mid, w1, _ in self.edges[src]:
                for dst, w2, _ in self.edges[mid]:
                    if dst == src:
                        continue
                    score = abs(w1 * w2)
                    hops.append((score, dst))
            hops.sort(key=lambda x: x[0], reverse=True)
            for _, dst in hops[:2]:
                activation[dst] = self._clamp(activation[dst] + jump_scale * src_act)

    def strongest_links(self, top_k=20):
        links = []
        for src in self.nodes:
            for dst, w, rel in self.edges[src]:
                if src < dst:
                    links.append((abs(w), src, dst, rel, w))
        links.sort(reverse=True)
        return links[:top_k]

    def edge_count(self):
        return sum(len(v) for v in self.edges.values())

    def export_json(self, path, layers=None, activation=None):
        payload = {
            "node_count": len(self.nodes),
            "directed_edge_count": self.edge_count(),
            "nodes": sorted(self.nodes),
            "edges": [],
        }

        for src in sorted(self.nodes):
            for dst, weight, relation in self.edges[src]:
                payload["edges"].append(
                    {
                        "source": src,
                        "target": dst,
                        "weight": round(weight, 6),
                        "relation": relation,
                    }
                )

        if layers is not None:
            payload["layers"] = layers
            payload["layer_sizes"] = [len(layer) for layer in layers]

        if activation is not None:
            payload["activation"] = {
                node: round(value, 6) for node, value in sorted(activation.items())
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def build_yellow_lemon_graph():
    g = ConceptGraph(seed=11)

    nodes = [
        "yellow", "lemon", "sour", "freshness", "sunlight", "summer", "cleanliness",
        "energy", "attention", "warning", "caution_sign", "gold", "wealth", "joy",
        "optimism", "memory", "childhood", "kitchen", "tea", "honey", "vitamin_c",
        "health", "immune_support", "zest", "fragrance", "bitter", "balance",
        "contrast", "creativity", "focus"
    ]

    for n in nodes:
        g.add_node(n)

    # Core sensory/cognitive cluster around yellow <-> lemon.
    g.connect("yellow", "lemon", 0.98, "associated_with")
    g.connect("lemon", "sour", 0.95, "sensory")
    g.connect("lemon", "fragrance", 0.88, "sensory")
    g.connect("lemon", "zest", 0.90, "part_of")
    g.connect("yellow", "sunlight", 0.92, "evokes")
    g.connect("sunlight", "summer", 0.86, "associated_with")
    g.connect("yellow", "attention", 0.83, "causes")
    g.connect("attention", "focus", 0.87, "causes")
    g.connect("yellow", "joy", 0.78, "evokes")
    g.connect("joy", "optimism", 0.90, "causes")

    # Freshness/health cluster.
    g.connect("lemon", "freshness", 0.93, "associated_with")
    g.connect("freshness", "cleanliness", 0.85, "cultural")
    g.connect("lemon", "vitamin_c", 0.91, "is_a", bidirectional=False)
    g.connect("vitamin_c", "health", 0.84, "causes")
    g.connect("health", "immune_support", 0.88, "causes")
    g.connect("lemon", "tea", 0.80, "cultural")
    g.connect("tea", "honey", 0.76, "associated_with")

    # Warning/value duality of yellow.
    g.connect("yellow", "warning", 0.82, "cultural")
    g.connect("warning", "caution_sign", 0.95, "is_a")
    g.connect("yellow", "gold", 0.79, "metaphor")
    g.connect("gold", "wealth", 0.90, "associated_with")

    # Contrast and balancing taste/cognition pathways.
    g.connect("sour", "bitter", 0.61, "associated_with")
    g.connect("sour", "contrast", 0.70, "metaphor")
    g.connect("contrast", "balance", 0.82, "causes")
    g.connect("balance", "creativity", 0.71, "causes")

    # Memory pathways.
    g.connect("lemon", "kitchen", 0.73, "associated_with")
    g.connect("kitchen", "childhood", 0.67, "evokes")
    g.connect("childhood", "memory", 0.92, "part_of")
    g.connect("memory", "focus", 0.62, "contrast")

    # Extra cross-links for richer graph traversal.
    g.connect("energy", "focus", 0.74, "causes")
    g.connect("sunlight", "energy", 0.81, "causes")
    g.connect("freshness", "energy", 0.72, "evokes")
    g.connect("optimism", "creativity", 0.77, "causes")
    g.connect("cleanliness", "health", 0.79, "associated_with")

    return g


def attach_decision_making_graph(g):
    """Bridge the idea graph into a decision-making subgraph.

    Existing concepts (attention, focus, balance, contrast, creativity, memory)
    become the inputs that a decision process needs, and the decision nodes
    converge on a single question: how do we make a good decision?
    """
    decision_nodes = [
        "good_decision", "good_outcome", "process_quality", "clear_criteria",
        "information_gathering", "diverse_perspectives", "pre_mortem", "bias_awareness",
        "overconfidence", "sunk_cost", "confirmation_bias", "reversibility",
        "two_way_door", "one_way_door", "expected_value", "risk_tolerance",
        "second_order_effects", "opportunity_cost", "decision_journal", "feedback_loop",
        "commitment", "decisiveness", "regret_minimization", "time_pressure", "clarity",
    ]
    for n in decision_nodes:
        g.add_node(n)

    # Bridges from the existing idea graph into decision-making inputs.
    g.connect("attention", "information_gathering", 0.80, "causes")
    g.connect("focus", "clear_criteria", 0.82, "causes")
    g.connect("balance", "risk_tolerance", 0.75, "associated_with")
    g.connect("contrast", "diverse_perspectives", 0.72, "metaphor")
    g.connect("creativity", "second_order_effects", 0.68, "evokes")
    g.connect("memory", "decision_journal", 0.70, "associated_with")

    # Evaluation chain: information and criteria feed expected-value reasoning.
    g.connect("information_gathering", "clear_criteria", 0.78, "causes")
    g.connect("clear_criteria", "expected_value", 0.85, "causes")
    g.connect("risk_tolerance", "expected_value", 0.70, "causes")
    g.connect("second_order_effects", "expected_value", 0.73, "causes")
    g.connect("opportunity_cost", "expected_value", 0.66, "causes")
    g.connect("expected_value", "good_decision", 0.90, "causes")

    # Bias-correction chain: exposure and pre-mortems suppress known failure modes.
    g.connect("diverse_perspectives", "bias_awareness", 0.86, "causes")
    g.connect("pre_mortem", "bias_awareness", 0.80, "causes")
    g.connect("pre_mortem", "second_order_effects", 0.66, "evokes")
    g.connect("bias_awareness", "overconfidence", -0.70, "inhibits", bidirectional=False)
    g.connect("bias_awareness", "confirmation_bias", -0.65, "inhibits", bidirectional=False)
    g.connect("bias_awareness", "sunk_cost", -0.60, "inhibits", bidirectional=False)
    g.connect("overconfidence", "good_decision", -0.55, "contrast", bidirectional=False)
    g.connect("sunk_cost", "good_decision", -0.60, "contrast", bidirectional=False)
    g.connect("confirmation_bias", "good_decision", -0.50, "contrast", bidirectional=False)

    # Process-over-outcome chain: decisions are judged by process, not just result.
    g.connect("decision_journal", "feedback_loop", 0.88, "causes")
    g.connect("feedback_loop", "process_quality", 0.85, "causes")
    g.connect("process_quality", "good_decision", 0.92, "causes")
    g.connect("good_decision", "good_outcome", 0.55, "associated_with")

    # Reversibility chain: door type sets how much deliberation a call deserves.
    g.connect("reversibility", "two_way_door", 0.90, "is_a", bidirectional=False)
    g.connect("reversibility", "one_way_door", -0.40, "contrast", bidirectional=False)
    g.connect("two_way_door", "decisiveness", 0.82, "causes")
    g.connect("one_way_door", "clear_criteria", 0.75, "causes")
    g.connect("one_way_door", "time_pressure", 0.40, "associated_with")
    g.connect("time_pressure", "clarity", -0.50, "contrast")
    g.connect("clarity", "clear_criteria", 0.70, "causes")

    # Closing the loop: regret minimization and commitment turn judgment into action.
    g.connect("regret_minimization", "good_decision", 0.60, "causes")
    g.connect("decisiveness", "commitment", 0.68, "causes")
    g.connect("commitment", "process_quality", 0.55, "causes")

    return decision_nodes


DECISION_ACTION_TIPS = {
    "information_gathering": "Gather enough information to see the shape of the problem, then stop before analysis paralysis sets in.",
    "diverse_perspectives": "Ask someone who disagrees with you before you decide, not after.",
    "pre_mortem": "Imagine the decision has already failed and write down why, before you commit.",
    "bias_awareness": "Name the bias you might be falling for out loud; naming it weakens it.",
    "clear_criteria": "Write down what 'good' looks like before you compare options, not after.",
    "expected_value": "Weigh outcomes by how likely and how big they are, not by how vivid they feel.",
    "reversibility": "Ask first: can this be undone? Reversible calls deserve speed, not deliberation.",
    "two_way_door": "Treat reversible decisions as cheap experiments, and move fast.",
    "one_way_door": "Slow down for irreversible decisions and demand real clarity first.",
    "decision_journal": "Write down the decision and your reasoning before you know the outcome.",
    "feedback_loop": "Review past decisions against their reasoning, not just their results.",
    "process_quality": "Judge yourself on whether the process was sound, even when the outcome was bad.",
    "regret_minimization": "Ask which choice you would regret more in ten years, not in ten minutes.",
    "opportunity_cost": "Remember every choice quietly cancels the next-best alternative.",
    "commitment": "Once a reversible decision is made, commit fully instead of re-litigating it.",
}


def decision_recommendations(activation, top_k=8):
    """Turn decision-node activation into a ranked, actionable checklist."""
    relevant = {
        k: v for k, v in activation.items() if k in DECISION_ACTION_TIPS
    }
    ranked = sorted(relevant.items(), key=lambda x: x[1], reverse=True)
    return [(name, score, DECISION_ACTION_TIPS[name]) for name, score in ranked[:top_k]]


def generate_decision_theories(activation):
    active = {k for k, v in activation.items() if v > 0.40}
    suppressed = {k for k, v in activation.items() if v < -0.15}
    theories = []

    if {"information_gathering", "clear_criteria", "expected_value"}.issubset(active):
        theories.append(
            "Structured Evaluation Theory: gathering information and fixing clear criteria upstream "
            "lets expected-value reasoning become the decisive lever, instead of gut instinct alone."
        )

    if {"diverse_perspectives", "bias_awareness"}.issubset(active):
        theories.append(
            "Bias Correction Theory: exposing a decision to diverse perspectives is the most reliable way "
            "to catch blind spots before they harden into commitments."
        )

    if {"decision_journal", "feedback_loop", "process_quality"}.issubset(active):
        theories.append(
            "Process Over Outcome Theory: recording decisions and reviewing them later builds the feedback "
            "loop that improves judgment over time, independent of any single outcome."
        )

    if {"reversibility", "two_way_door", "decisiveness"}.issubset(active):
        theories.append(
            "Reversibility Speed Theory: reversible decisions should be made quickly and cheaply, freeing "
            "deliberation time for the rare irreversible, one-way-door choices."
        )

    if "bias_awareness" in active and suppressed & {"overconfidence", "sunk_cost", "confirmation_bias"}:
        theories.append(
            "Debiasing Payoff Theory: actively suppressing overconfidence, sunk-cost thinking, and "
            "confirmation bias raises decision quality more reliably than acquiring more information."
        )

    if not theories:
        theories.append(
            "Decision Web Theory: attention, balance, and contrast from the idea graph feed directly into "
            "criteria-setting and bias-checking, so a good decision is the outcome of a well-connected process."
        )

    return theories


def attach_exponential_layers(
    g,
    seed_nodes,
    depth=4,
    branch_factor=2,
    cluster_size=2,
    seed=23,
):
    """Attach explicit exponential branching layers to a concept graph.

    Growth factor per layer is approximately: branch_factor * cluster_size.
    This remains non-neural: it is graph construction + activation diffusion.
    """
    rng = random.Random(seed)
    relation_pool = ["associated_with", "evokes", "causes", "metaphor", "cultural", "sensory"]
    domains = [
        "signal", "chemistry", "memory", "emotion", "language",
        "design", "ritual", "attention", "value", "contrast"
    ]

    layers = [list(seed_nodes)]

    for level in range(1, depth + 1):
        prev = layers[-1]
        current = []

        for parent in prev:
            for branch in range(branch_factor):
                domain = domains[(hash((parent, level, branch)) % len(domains))]
                cluster = []

                for idx in range(cluster_size):
                    child = f"L{level}_{domain}_{branch}_{idx}_from_{parent}"
                    g.add_node(child)
                    cluster.append(child)
                    rel = rng.choice(relation_pool)
                    weight = rng.uniform(0.58, 0.96)
                    g.connect(parent, child, weight, rel, bidirectional=True)

                # Make each new cluster a locally coherent "different bunch".
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        rel = rng.choice(["associated_with", "contrast", "part_of"])
                        weight = rng.uniform(0.40, 0.82)
                        g.connect(cluster[i], cluster[j], weight, rel, bidirectional=True)

                current.extend(cluster)

        # Inhibitory cross-layer links add competition between expanding branches.
        inhibition_count = max(1, len(current) // 10)
        for _ in range(inhibition_count):
            src = rng.choice(current)
            dst = rng.choice(prev)
            g.connect(src, dst, -rng.uniform(0.25, 0.62), "inhibits", bidirectional=False)

        # Add sparse skip links to an older layer to improve long-distance propagation.
        if level >= 2:
            two_back = layers[level - 2]
            link_budget = max(1, len(current) // 8)
            for _ in range(link_budget):
                a = rng.choice(current)
                b = rng.choice(two_back)
                g.connect(a, b, rng.uniform(0.35, 0.70), "metaphor", bidirectional=True)

            # Additional inhibition to two-back layers for selective gating.
            inhibit_skip = max(1, len(current) // 16)
            for _ in range(inhibit_skip):
                a = rng.choice(current)
                b = rng.choice(two_back)
                g.connect(a, b, -rng.uniform(0.20, 0.50), "inhibits", bidirectional=False)

        layers.append(current)

    return layers


def layer_growth_report(layers):
    counts = [len(layer) for layer in layers]
    multipliers = []
    for i in range(1, len(counts)):
        prev = max(1, counts[i - 1])
        multipliers.append(counts[i] / prev)
    return counts, multipliers


def top_activated(activation, top_k=12):
    ordered = sorted(activation.items(), key=lambda x: abs(x[1]), reverse=True)
    return ordered[:top_k]


def generate_theories(activation):
    active = {k for k, v in activation.items() if v > 0.45}
    theories = []

    if {"yellow", "attention", "focus"}.issubset(active):
        theories.append(
            "Signal Theory: Yellow functions as an attentional primer, and lemon-related cues convert alertness into sustained focus."
        )

    if {"lemon", "freshness", "cleanliness", "health"}.issubset(active):
        theories.append(
            "Purity Chain Theory: Lemon maps sensory freshness into a cultural model of cleanliness, which then projects into health expectations."
        )

    if {"sunlight", "joy", "optimism", "creativity"}.issubset(active):
        theories.append(
            "Mood-to-Generation Theory: Sunlight-yellow associations elevate mood, and elevated mood increases creative association breadth."
        )

    if {"sour", "contrast", "balance"}.issubset(active):
        theories.append(
            "Contrast-Balance Theory: Sourness acts as a contrast trigger; controlled contrast helps cognitive systems search for balance and better ideas."
        )

    if {"kitchen", "childhood", "memory"}.issubset(active):
        theories.append(
            "Episodic Anchor Theory: Lemon cues anchor in domestic episodic memory, making related ideas easier to retrieve under ambiguity."
        )

    if not theories:
        theories.append(
            "Association Field Theory: Yellow-lemon behaves as a high-connectivity seed pair that spreads into sensory, emotional, and cultural subgraphs."
        )

    return theories


def main():
    graph = build_yellow_lemon_graph()

    # Bridge the idea graph into a decision-making subgraph before expansion,
    # so the two keep sharing structure rather than growing in isolation.
    decision_nodes = attach_decision_making_graph(graph)

    # Explicit exponential expansion: each layer branches into new concept clusters.
    layers = attach_exponential_layers(
        graph,
        seed_nodes=["yellow", "lemon"],
        depth=4,
        branch_factor=2,
        cluster_size=2,
        seed=23,
    )
    layer_counts, multipliers = layer_growth_report(layers)

    seeds = {
        "yellow": 1.0,
        "lemon": 1.0,
        "clear_criteria": 0.6,
        "diverse_perspectives": 0.6,
        "decision_journal": 0.5,
        "pre_mortem": 0.5,
        "reversibility": 0.5,
    }
    activation = graph.diffuse(
        seeds,
        steps=8,
        decay=0.22,
        tunnel_threshold=0.008,
        target_rms=0.42,
        soft_cap=0.92,
    )
    graph.export_json("concept_graph_export.json", layers=layers, activation=activation)

    print("NON-NEURAL EXPANDING CONCEPT GRAPH")
    print("Seed concepts:", seeds)
    print("Layer sizes:", layer_counts)
    print("Layer multipliers:", [round(x, 2) for x in multipliers])
    print("Total nodes:", len(graph.nodes), "Total directed edges:", graph.edge_count())
    print("Exported graph JSON: concept_graph_export.json")
    print("\nTop activated concepts:")
    for name, score in top_activated(activation, top_k=14):
        print(f"  {name:15s} {score:+.4f}")

    print("\nStrongest structural links:")
    for _, src, dst, rel, w in graph.strongest_links(top_k=14):
        print(f"  {src:12s} <-> {dst:12s}  w={w:+.2f}  rel={rel}")

    print("\nGenerated connection theories:")
    for i, t in enumerate(generate_theories(activation), start=1):
        print(f"  {i}. {t}")

    decision_activation = {k: v for k, v in activation.items() if k in decision_nodes}
    print("\nDecision-making activation (how to make a good decision):")
    for name, score in top_activated(decision_activation, top_k=len(decision_nodes)):
        print(f"  {name:20s} {score:+.4f}")

    print("\nGenerated decision-making theories:")
    for i, t in enumerate(generate_decision_theories(activation), start=1):
        print(f"  {i}. {t}")

    print("\nActionable checklist, ranked by current activation:")
    for i, (name, score, tip) in enumerate(decision_recommendations(activation), start=1):
        print(f"  {i}. [{name}, {score:+.2f}] {tip}")


if __name__ == "__main__":
    main()
