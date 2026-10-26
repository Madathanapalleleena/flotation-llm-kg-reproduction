"""Consistency checks for the assembled knowledge graph."""
from ontology.knowledge_tree import KnowledgeTree
from kg.triple_validator import count_conflicts


def graph_consistency_assessment(graph, kt=None):
    """Report duplicate IDs, invalid node schemas, dangling edges, and unknown relation types."""
    kt = kt or KnowledgeTree()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    ids = [node.get("id") for node in nodes]
    id_set = {node_id for node_id in ids if node_id}
    duplicate_node_ids = len(ids) - len(id_set)

    valid_kinds = {"Ontology", "Entity", "Image"}
    valid_entity_types = set(kt.entity_types)
    valid_ontology_labels = {node.cls for node in kt.nodes.values()}
    required_node_properties = kt.graph_schema.get("node_required_properties", ["id", "name", "kind", "type"])
    node_property_types = kt.graph_schema.get("node_property_types", {})
    invalid_nodes = 0
    for node in nodes:
        kind = node.get("kind")
        valid = all(node.get(field) is not None for field in required_node_properties) and kind in valid_kinds
        for field, expected in node_property_types.items():
            value = node.get(field)
            if value is not None and expected == "string" and not isinstance(value, str):
                valid = False
        if kind == "Entity":
            valid = valid and node.get("type") in valid_entity_types
        elif kind == "Ontology":
            valid = valid and node.get("type") in valid_ontology_labels
        elif kind == "Image":
            valid = valid and node.get("type") == "Image" and bool(node.get("url"))
        invalid_nodes += not valid

    valid_relations = {relation.upper() for relation in kt.relation_types}
    valid_relations.update(edge[1].upper() for edge in kt.edges)
    valid_relations.update({"HAS_ENTITIES", "HAS_IMG"})
    dangling_edges = 0
    invalid_relationship_types = 0
    duplicate_relationships = 0
    invalid_relationships = 0
    relationship_keys = set()
    required_relationship_properties = kt.graph_schema.get(
        "relationship_required_properties", ["source", "target", "type"])
    relationship_property_types = kt.graph_schema.get("relationship_property_types", {})
    entity_names = {node.get("id"): node.get("name", "") for node in nodes}
    graph_triples = []
    for edge in edges:
        dangling_edges += edge.get("source") not in id_set or edge.get("target") not in id_set
        invalid_relationship_types += str(edge.get("type", "")).upper() not in valid_relations
        invalid_relationships += not all(edge.get(field) is not None for field in required_relationship_properties)
        key = (edge.get("source"), edge.get("target"), edge.get("type"))
        duplicate_relationships += key in relationship_keys
        relationship_keys.add(key)
        for field, expected in relationship_property_types.items():
            value = edge.get(field)
            if value is None:
                continue
            if expected == "string" and not isinstance(value, str):
                invalid_relationships += 1
            elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                invalid_relationships += 1
            elif expected == "string_list" and (
                    not isinstance(value, list) or any(not isinstance(item, str) for item in value)):
                invalid_relationships += 1
            elif expected == "integer_list" and (
                    not isinstance(value, list) or any(not isinstance(item, int) or isinstance(item, bool)
                                                        for item in value)):
                invalid_relationships += 1
        if edge.get("kind") == "triple":
            graph_triples.append({
                "subject": entity_names.get(edge.get("source"), ""),
                "object": entity_names.get(edge.get("target"), ""),
                "relation": str(edge.get("type", "")).lower(),
            })

    contradictory_triples = count_conflicts(graph_triples)
    issues = (duplicate_node_ids + invalid_nodes + dangling_edges + invalid_relationship_types
              + invalid_relationships + duplicate_relationships + contradictory_triples)
    total = len(nodes) + len(edges)
    return {
        "n_nodes": len(nodes),
        "n_relationships": len(edges),
        "duplicate_node_ids": duplicate_node_ids,
        "invalid_nodes": invalid_nodes,
        "dangling_relationships": dangling_edges,
        "unknown_relationship_types": invalid_relationship_types,
        "invalid_relationships": invalid_relationships,
        "duplicate_relationships": duplicate_relationships,
        "contradictory_triples": contradictory_triples,
        "n_issues": issues,
        "consistency": 1.0 - min(issues, total) / max(total, 1),
    }
