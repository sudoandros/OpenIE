import argparse
import xml.etree.ElementTree as ET

def fix_gexf(root_element):
    graph_node = root_element.find("{http://www.gexf.net/1.1draft}graph")
    attributes_nodes = graph_node.findall("{http://www.gexf.net/1.1draft}attributes")
    edge_attributes = {}
    node_attributes = {}
    for attributes_node in attributes_nodes:
        for attribute_node in attributes_node.findall(
            "{http://www.gexf.net/1.1draft}attribute"
        ):
            attr_id = attribute_node.get("id")
            attr_title = attribute_node.get("title")
            attribute_node.set("id", attr_title)
            if attributes_node.get("class") == "edge":
                edge_attributes[attr_id] = attr_title
            elif attributes_node.get("class") == "node":
                node_attributes[attr_id] = attr_title
    nodes_node = graph_node.find("{http://www.gexf.net/1.1draft}nodes")
    for node_node in nodes_node.findall("{http://www.gexf.net/1.1draft}node"):
        for attvalue_node in node_node.find(
            "{http://www.gexf.net/1.1draft}attvalues"
        ).findall("{http://www.gexf.net/1.1draft}attvalue"):
            for_value = attvalue_node.get("for")
            attvalue_node.set("for", node_attributes[for_value])
    edges_node = graph_node.find("{http://www.gexf.net/1.1draft}edges")
    for edge_node in edges_node.findall("{http://www.gexf.net/1.1draft}edge"):
        for attvalue_node in edge_node.find(
            "{http://www.gexf.net/1.1draft}attvalues"
        ).findall("{http://www.gexf.net/1.1draft}attvalue"):
            for_value = attvalue_node.get("for")
            attvalue_node.set("for", edge_attributes[for_value])
    return root_element

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gexf_path", help="Path to the gexf file")
    parser.add_argument("out_path", help="Path to the directory to save result gexf to")
    args = parser.parse_args()

    with open(args.gexf_path, mode="r", encoding="utf-8") as xml_file:
        new_root = fix_gexf(ET.fromstring(xml_file.read()))
    ET.register_namespace("", "http://www.gexf.net/1.1draft")
    new_xml = ET.ElementTree(new_root)
    new_xml.write(args.out_path, encoding="utf-8")
